import argparse
import os
import sys
import time
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import PermutationInvariantTraining
import glob
import numpy as np
import itertools
from PMSimCLR_loss import ManifoldSimCLRLoss
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

sys.path.append('/n/home11/nswood/weaver-core')
from weaver.nn.model.PMNN import PMNN
from weaver.nn.model.PM_utils import ManifoldNNLayer
import geoopt

from scipy.optimize import linear_sum_assignment



class ParticleCLRModel(nn.Module):
    def __init__(self, input_dim, projection_dim, embedder_model, k = 0, learnable = True):
        super(ParticleCLRModel, self).__init__()
        self.embedder_model = embedder_model
        self.man = geoopt.Stereographic(k=k, learnable=learnable)   
        
        

    
    def forward(self, x):
        # x: shape (B, C, N)
        # Assume embedder_model returns (B, D, N); we transpose to (B, N, D)
        B, C, N = x.shape
        # particle_emb = self.embedder_model(x, embed_parts=True)
        
        x = x.permute(0, 2, 1)

        x = self.man.proju(self.man.origin(x.shape), x)
        x = self.man.expmap0(x, project=True)

        particle_emb = self.embedder_model(x)

        if isinstance(particle_emb, tuple):
            particle_emb = particle_emb[1]

        # particle_emb = particle_emb.transpose(1, 2)  # (B, N, D)
        if self.man.name != 'Euclidean':
            agg = self.man.weighted_midpoint(particle_emb, dim = 0,reducedim=[1],keepdim = True)
            agg = self.man.logmap0(agg)
            # particle_emb = self.man.mobius_add(agg, particle_emb)  
        else: 
            agg = torch.mean(particle_emb, dim=1,keepdim=True)
            # particle_emb = agg + particle_emb
        
        agg = agg.repeat(1, particle_emb.shape[1], 1)
        particle_emb = self.man.logmap0(particle_emb)
        particle_emb = torch.cat((particle_emb, agg), dim = -1)

        return particle_emb
from typing import List, Optional

def list_range(end: int):
    res: List[int] = []
    for d in range(end):
        res.append(d)
    return res

def load_data_from_globs(glob_paths):
        all_data = []
        all_labels = []
        for data_file in glob_paths:
            # print(f'Loading {data_file}...')
            with h5py.File(data_file, 'r') as cur_file:
                # Directly convert the dataset to torch tensor to avoid extra numpy concatenation
                cur_data = cur_file['datasets'][:]
                cur_labels = cur_file['labels'][:]
                # print('Cur Data:', cur_data.shape)
                # print('Cur Labels:', cur_labels.shape)
                new_data = []
                new_labels = []
                for i in range(cur_labels.shape[0]):
                    sample_data = []
                    sample_labels = []
                    for lbl in [0, 1, 2, 3]:
                        idx = np.where(cur_labels[i] == lbl)[0][:100]
                        sample_data.append(cur_data[i, idx, :])
                        sample_labels.append(cur_labels[i, idx])
                    sample_data = np.concatenate(sample_data, axis=0)
                    sample_labels = np.concatenate(sample_labels, axis=0)
                    
                    perm = np.random.permutation(sample_data.shape[0])
                    sample_data = sample_data[perm]
                    sample_labels = sample_labels[perm]

                    new_data.append(sample_data)
                    new_labels.append(sample_labels)
                cur_data = np.stack(new_data, axis=0)
                cur_labels = np.stack(new_labels, axis=0)
                # print('Cur Data:', cur_data.shape)
                # print('Cur Labels:', cur_labels.shape)
                unique_labels, label_counts = np.unique(cur_labels, return_counts=True)
                # print("Label counts:", dict(zip(unique_labels, label_counts)))
                all_data.append(cur_data)
                all_labels.append(cur_labels)

        data_array = np.concatenate(all_data)
        labels_array = np.concatenate(all_labels)
        all_data = torch.tensor(data_array)
        all_labels = torch.tensor(labels_array)

        return all_data, all_labels 

def supervised_contrastive_loss(embeddings, labels, temperature=0.01, eps=1e-9, sim_metric='cos'):
    """
    Vectorized supervised contrastive loss.

    embeddings: Tensor of shape (B, N, D) where B is batch size, N is the number of particles, and D is embedding dim.
    labels: Tensor of shape (B, N) with integer class labels.
    temperature: Scaling factor for similarity scores.
    eps: A small value to avoid log(0).
    sim_metric: 'dist' to use negative Euclidean distance as similarity, 'cos' to use cosine similarity.
    """
    B, N, D = embeddings.shape
    device = embeddings.device

    # Compute similarity matrix based on the chosen metric.
    if sim_metric == 'cos':
        # Compute cosine similarity; assume embeddings are normalized.
        sim_matrix = torch.bmm(embeddings, embeddings.transpose(1, 2)) / temperature
    elif sim_metric == 'dist':
        # Compute pairwise Euclidean distances and convert them to similarity scores.
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)  # (B, N, N)
        sim_matrix = -distance_matrix / temperature  # Invert distances to get similarity scores
    else:
        raise ValueError("sim_metric must be either 'dist' or 'cos'.")

    # Create a mask to zero out self-similarities (diagonals) for each cloud.
    diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
    sim_matrix = sim_matrix.masked_fill(diag_mask, -1e9)

    # Create a positive mask: for each cloud, positives[i,j] is True if labels[i]==labels[j]
    labels_expanded_i = labels.unsqueeze(2)  # (B, N, 1)
    labels_expanded_j = labels.unsqueeze(1)  # (B, 1, N)
    positive_mask = (labels_expanded_i == labels_expanded_j)  # (B, N, N)
    # Remove self comparisons from positive mask.
    positive_mask = positive_mask & (~diag_mask)

    # Exponentiate the similarity scores.
    exp_sim = torch.exp(sim_matrix)  # (B, N, N)

    # For each anchor, sum over positives and sum over all (non-self) similarities.
    numerator = (exp_sim * positive_mask.float()).sum(dim=2)  # (B, N)
    denominator = exp_sim.sum(dim=2)  # (B, N)

    # Compute the loss per anchor: -log(positive_sum / denominator)
    loss_per_anchor = -torch.log(numerator / (denominator + eps) + eps)  # (B, N)

    # Only consider anchors that have at least one positive; create a valid mask.
    valid_mask = (positive_mask.sum(dim=2) > 0).float()  # (B, N)
    # Sum the loss over all valid anchors and average.
    loss = (loss_per_anchor * valid_mask).sum() / (valid_mask.sum() + eps)
    
    return loss



# Skeleton function for loading a model.
def load_model(model_name, device, part_geom, part_dim, k, learnable):
    k = float(k)
    if model_name == 'test':
        if type(part_dim) == tuple:
            part_dim = part_dim[0]
        elif type(part_dim) == str:
            part_dim = int(part_dim)
        
        if part_dim is None:
            part_dim = 2
        # Example simple model
        embedder = nn.Sequential(
            nn.Linear(2, part_dim),
            nn.ReLU(),
            nn.Linear(part_dim, part_dim)
        )
        man = geoopt.Euclidean()
    else:
        print('Building PM-MLP model')
        print('part_geom:', part_geom)
        print('part_dim:', part_dim)
        print('part_curvature_init:', k)
        print('part_curvature_trainable:', learnable)
        # Default model
        if part_geom == 'R':
            man = geoopt.Euclidean()    
            learnable = False
        elif part_geom == 'H':
            man = geoopt.PoincareBallExact(k=float(k), learnable=learnable)
        elif part_geom == 'S':
            man = geoopt.SphereProjectionExact(k=float(k), learnable=learnable)
        elif part_geom == 'M':
            man = geoopt.StereographicExact(k=float(k), learnable=learnable)
        else:
            raise ValueError(f"Unsupported part_geom: {part_geom}")
        if type(part_dim) == str:
            part_dim = int(part_dim)
        embedder = nn.Sequential(
            
            ManifoldNNLayer(2, part_dim, float(k),learnable, 0, nn.ReLU(),True),
            ManifoldNNLayer(part_dim, part_dim,float(k),learnable, 0, None,True)
        )
        
    input_dim= int(part_dim) 
    output_dim = 2 
    model = ParticleCLRModel(input_dim, output_dim, embedder,k = k, learnable = learnable)
    proj_model = nn.Sequential(
            nn.Linear(2*input_dim,4*output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(4*output_dim, output_dim)
        )
    return model.to(device).double(), proj_model.double()

def main():
    parser = argparse.ArgumentParser(
        description="Training script for Gaussian clustering using manifold representations"
    )
    parser.add_argument('--data_dir', type=str, 
                        help="Path to directory containing 'train', 'test', and 'val' subdirectories with h5 files.", default = "/n/holystore01/LABS/iaifi_lab/Lab/nswood/testing_hyperbolic_gaussians_toy")
    parser.add_argument('--outdir', type=str,
                        help="Directory to store training run outputs.", default = 'testing_manifold_gaussians')
    parser.add_argument('--model_name', type=str, default = 'model',
                        help="Model name to use (e.g., 'simple').")
    parser.add_argument('--part_geom', type=str, default='R',
                        help="Particle representation geometry")
    parser.add_argument('--part_dim', type=str, default='2',
                        help="Particle representation dimension")
    parser.add_argument('--part_curvature_init', type=str, default='-1',
                        help="Particle representation curvature initialization")
    parser.add_argument('--part_curvature_trainable', type=bool, default=True,
                        help="Particle representation curvature trainable")
    parser.add_argument('--batch_size', type=int, default=25,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument('--test_run', type=bool, default=False,
                        help="test_run.")
    args = parser.parse_args()

    # Create a new run directory under the provided outdir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.outdir, f'run_{args.model_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset from an h5 file
    

    # Load the dataset from the provided data directory
    if args.test_run:
        max_files = 10
    else:
        max_files = np.inf

    train_pattern = os.path.join(args.data_dir, 'train', '*.h5')
    train_files = glob.glob(train_pattern)
    test_pattern = os.path.join(args.data_dir, 'test', '*.h5')
    test_files = glob.glob(test_pattern)
    val_pattern = os.path.join(args.data_dir, 'val', '*.h5')
    val_files = glob.glob(val_pattern)

    if len(train_files) > max_files:
        train_files = train_files[:max_files]
    if len(test_files) > max_files:
        test_files = test_files[:max_files]
    if len(val_files) > max_files:
        val_files = val_files[:max_files]

    print('Loading train dataset...')
    train_data, train_labels = load_data_from_globs(train_files)
    print(f'Loaded {len(train_data)} samples')


    print('Loading test dataset...')
    test_data, test_labels = load_data_from_globs(test_files)
    print(f'Loaded {len(test_data)} samples')

    
    print('Loading val dataset...')
    val_data, val_labels = load_data_from_globs(val_files)
    print(f'Loaded {len(val_data)} samples')

            
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # --- Supervised Contrastive Loss (CLR) ---
    

    # --- Training Setup ---
    # Example: load the model (this function should return a ParticleCLRModel instance)
    model, projection_model = load_model(args.model_name, device, args.part_geom, args.part_dim, args.part_curvature_init, args.part_curvature_trainable)
    
    model = model.to(device)
    projection_model = projection_model.to(device) 


    num_params = sum(p.numel() for p in model.parameters())
    num_params += sum(p.numel() for p in projection_model.parameters())
    print("Model parameters count:", num_params)
    params_file = os.path.join(run_dir, "model_parameters.txt")
    with open(params_file, "w") as f:
        f.write(f"Number of model parameters: {num_params}\n")

    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)

    proj_optimizer = torch.optim.Adam(projection_model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler_proj = optim.lr_scheduler.ExponentialLR(proj_optimizer, gamma=0.95)


    
    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        projection_model.train()  # Ensure projection_model is in train mode as well.
        running_loss = 0.0
        if args.test_run:
            start_time = time.time()
        for batch_inputs, batch_labels in train_loader:
            # Ensure inputs are floats and labels are longs
            inputs = batch_inputs.double().to(device)
            labels = batch_labels.long().to(device)  # shape: (B, N)

            # Prepare inputs: shape (B, N, F) -> (B, F, N)
            B, N, F = inputs.shape
            inputs = inputs.permute(0, 2, 1)

            # Zero gradients for both optimizers
            optimizer.zero_grad()
            proj_optimizer.zero_grad()

            # Forward pass through embedding model then projection model
            embed = model(inputs)                     # embed shape: (B, N, projection_dim)
            outputs = projection_model(embed)         # outputs shape: (B, N, projection_dim) or similar

            # Compute supervised contrastive loss using labels
            loss = supervised_contrastive_loss(outputs, labels)
            loss.backward()

            # Update weights for both models
            optimizer.step()
            proj_optimizer.step()

            running_loss += loss.item()
        
        if args.test_run:
            epoch_time = time.time() - start_time
            print(f'Epoch: {epoch_time:.4f} seconds')
        epoch_loss = running_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        projection_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                inputs = batch_inputs.double().to(device)
                labels = batch_labels.long().to(device)  # shape: (B, N)
                B, N, F = inputs.shape
                inputs = inputs.permute(0, 2, 1)
                embed = model(inputs)
                outputs = projection_model(embed)
                loss = supervised_contrastive_loss(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        with open(os.path.join(run_dir, "log.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}\n")

        # Step the learning rate schedulers for both optimizers
        scheduler.step()
        scheduler_proj.step()

    # --- Testing Phase ---
    model.eval()
    projection_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            inputs = batch_inputs.double().to(device)
            labels = batch_labels.long().to(device)
            B, N, F = inputs.shape
            inputs = inputs.permute(0, 2, 1)
            embed = model(inputs)
            outputs = projection_model(embed)
            loss = supervised_contrastive_loss(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    import matplotlib.pyplot as plt

    # Plot embeddings for five test samples
    model.eval()
    # Get one batch from the test loader
    batch_inputs, batch_labels = next(iter(test_loader))
    batch_inputs = batch_inputs.double().to(device)
    batch_labels = batch_labels.long().to(device)
    B, N, F = batch_inputs.shape
    inputs_transposed = batch_inputs.permute(0, 2, 1)
    embeddings = model(inputs_transposed)  # Shape: (B, N, projection_dim)
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = batch_labels.cpu().detach().numpy()

    for i in range(5):
        emb_sample = embeddings_np[i]      # (N, 2) expected
        label_sample = labels_np[i]          # (N,)
        
        plt.figure()
        scatter = plt.scatter(emb_sample[:, 0], emb_sample[:, 1], c=label_sample, cmap='viridis', s=10)
        plt.colorbar(scatter)
        plt.title(f"Contrastive Embedding - Sample {i+1}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        
        plot_file = os.path.join(run_dir, f"embedding_sample_{i+1}.png")
        plt.savefig(plot_file)
        plt.close()
    # Save the final model state
    model_file = os.path.join(run_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Training complete. Model saved to {model_file}")

if __name__ == '__main__':
    main()