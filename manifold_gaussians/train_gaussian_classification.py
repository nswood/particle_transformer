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
import geoopt
sys.path.append('/n/home11/nswood/weaver-core')
from weaver.nn.model.PMNN import PMNN
from weaver.nn.model.PM_utils import ManifoldNNLayer

class PermutationInvariantLossVectorized(nn.Module):
    def __init__(self, base_loss_fn, num_classes=4):
        """
        Args:
            base_loss_fn (callable): A function that takes (pred, target) and returns a tensor
                                     of shape (batch_size*num_perms,) containing per-sample losses.
                                     The inputs should have shape (batch_size*num_perms, num_points, num_classes).
            num_classes (int): Number of classes.
        """
        super(PermutationInvariantLossVectorized, self).__init__()
        self.base_loss_fn = base_loss_fn
        perms = list(itertools.permutations(range(num_classes)))
        self.num_perms = len(perms)
        # Register the permutation tensor as a buffer so it moves with the model.
        self.register_buffer('perm_tensor', torch.tensor(perms))

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predictions with shape (batch_size, num_points, num_classes)
            y_true (torch.Tensor): Ground truth with shape (batch_size, num_points, num_classes)
        Returns:
            tuple: (averaged loss over the batch after selecting the best permutation, accuracy)
        """
        batch_size, num_points, num_classes = y_pred.shape
        if num_classes != self.perm_tensor.shape[1]:
            raise ValueError("Mismatch in number of classes between predictions and permutation tensor.")

        # Expand y_pred to shape: (batch_size, num_perms, num_points, num_classes)
        y_pred_expanded = y_pred.unsqueeze(1).expand(batch_size, self.num_perms, num_points, num_classes)
        # Expand the permutation tensor: shape (batch_size, num_perms, num_points, num_classes)
        perm_tensor_expanded = self.perm_tensor.view(1, self.num_perms, 1, num_classes).expand(batch_size, self.num_perms, num_points, num_classes)
        # Permute the class dimension for all samples.
        y_pred_permuted = torch.gather(y_pred_expanded, dim=3, index=perm_tensor_expanded)
        # Expand y_true to shape: (batch_size, num_perms, num_points, num_classes)
        y_true_expanded = y_true.unsqueeze(1).expand(batch_size, self.num_perms, num_points, num_classes)

        # Compute loss per sample per permutation.
        losses = self.base_loss_fn(y_pred_permuted.permute(0, 3, 2, 1), y_true_expanded.permute(0, 3, 2, 1).float())
        losses = torch.sum(losses, dim=1)
        losses = losses.view(batch_size, self.num_perms)

        # For each sample, find the permutation with the minimal loss.
        best_loss_values, best_perm_idx = losses.min(dim=1)
        loss_value = best_loss_values.mean()

        # Compute accuracy using predictions corresponding to the best permutation.
        # Get the best predictions: shape (batch_size, num_points, num_classes)
        best_y_pred = y_pred_permuted[torch.arange(batch_size), best_perm_idx, :, :]
        best_preds = best_y_pred.argmax(dim=2)
        y_true_labels = y_true.argmax(dim=2)
        # Calculate overall accuracy across the batch.
        correct = (best_preds == y_true_labels).float().mean()

        return loss_value, correct

class cluster_classifier_model(nn.Module):
    def __init__(self, input_dim, output_dim, embedder_model, man = None):
        super(cluster_classifier_model, self).__init__()
        self.embedder_model = embedder_model

        # self.agg_model = nn.Sequential(
        #     nn.Linear(400*input_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 24)
        # )
        self.layernorm = nn.LayerNorm(input_dim)
        self.man = man
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
    def forward(self, x):
        B,F,N = x.shape
        # print('x:', x.shape)
        # tan_x = self.embedder_model(x,embed_parts = True)


        x = x.permute(0,2,1)
        
        if self.man is not None:
            x = self.man.expmap0(x)
            x = self.embedder_model(x)
            # tan_x = self.man.logmap0(x)
        else:
            tan_x = self.embedder_model(x)

        
        if type(tan_x) == tuple:
            tan_x = tan_x[1]
        # print('tan_x:', tan_x.shape)
        tan_x = tan_x.view(B, -1)
        # print('tan_x:', tan_x.shape)
       
        tan_x = self.layernorm(tan_x)
        x = self.classifier(tan_x)
        
        x = x.view(B, N, -1)
        # print('x:', x.shape)
        return x

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
                        idx = np.where(cur_labels[i] == lbl)[0][:10]
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
                # print("Label counts:", dict(zip(unique_labels, label_counts)))
                all_data.append(cur_data)
                all_labels.append(cur_labels)

        data_array = np.concatenate(all_data)
        labels_array = np.concatenate(all_labels)
        all_data = torch.tensor(data_array)
        all_labels = torch.tensor(labels_array)
        all_labels = torch.nn.functional.one_hot(all_labels.long(), num_classes=4)
        
        return all_data, all_labels 



# Skeleton function for loading a model.
def load_model(model_name, device, part_geom, part_dim, part_curvature_init, part_curvature_trainable):
    if model_name == 'test':
        if type(part_dim) == tuple:
            part_dim = part_dim[0]
        elif type(part_dim) == str:
            part_dim = int(part_dim)
        
        print('Building test model')
        print('part_dim:', part_dim)
        # print('type of part_dim:', type(part_dim))

        if part_dim is None:
            part_dim = 2
        # Example simple model
        embedder = nn.Sequential(
            nn.Linear(2, part_dim),
            nn.ReLU(),
            nn.Linear(part_dim, part_dim)
        )
        man = None
    else:
        print('Building PM-MLP model')
        print('part_geom:', part_geom)
        print('part_dim:', part_dim)
        print('part_curvature_init:', part_curvature_init)
        print('part_curvature_trainable:', part_curvature_trainable)
        # Default model
        
        if part_geom == 'R':
            man = geoopt.Euclidean()
        elif part_geom == 'H':
            man = geoopt.PoincareBallExact(c=float(part_curvature_init), learnable=part_curvature_trainable)
        elif part_geom == 'S':
            man = geoopt.SphereProjectionExact(k=float(part_curvature_init), learnable=part_curvature_trainable)
        elif part_geom == 'M':
            man = geoopt.StereographicExact(k=float(part_curvature_init), learnable=part_curvature_trainable)
        else:
            raise ValueError(f"Unsupported part_geom: {part_geom}")
        if type(part_dim) == str:
            part_dim = int(part_dim)
        embedder = nn.Sequential(
            ManifoldNNLayer(2, part_dim, man, man.k, 0, nn.ReLU(),True),
            ManifoldNNLayer(part_dim, part_dim, man, man.k, 0, None,True)
        )
        
        # embedder = PMNN(2,
        #     part_geom = part_geom,
        #     part_dim =  part_dim,
        #     part_curvature_init = part_curvature_init,
        #     learnable = part_curvature_trainable)
        
    input_dim= int(part_dim)*40
    output_dim = 40*4
    if man is not None:
        model = cluster_classifier_model(input_dim, output_dim, embedder)
    else:
        model = cluster_classifier_model(input_dim, output_dim,embedder, man = man)
    return model.to(device).double()

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
    parser.add_argument('--part_curvature_trainable', type=bool, default=False,
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
    model = load_model(args.model_name, device, args.part_geom, args.part_dim, 
                    args.part_curvature_init, args.part_curvature_trainable)
    # For CLR, ensure your loaded model is of type ParticleCLRModel (or update accordingly)
    num_params = sum(p.numel() for p in model.parameters())
    print("Model parameters count:", num_params)
    params_file = os.path.join(run_dir, "model_parameters.txt")
    with open(params_file, "w") as f:
        f.write(f"Number of model parameters: {num_params}\n")

    # Here we no longer use cross entropy; we use the contrastive loss.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    base_loss = nn.CrossEntropyLoss(reduction='none')
    criterion = PermutationInvariantLossVectorized(base_loss, num_classes=4).to(device)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        if args.test_run:
            start_time = time.time()
        for batch_inputs, batch_labels in train_loader:
            # Ensure inputs are floats and labels are longs
            inputs = batch_inputs.double().to(device)
            # For contrastive loss, we use label indices directly.
            labels = batch_labels.long().to(device)  # shape: (B, N)
            
            # Prepare inputs: shape (B, N, F) -> (B, F, N)
            B, N, F = inputs.shape
            inputs = inputs.permute(0, 2, 1)

            optimizer.zero_grad()
            # Model outputs: per-particle embeddings (B, N, projection_dim)
            outputs = model(inputs)
            # Compute CLR loss using supervised pairs based on labels.
            loss, acc = criterion(outputs, labels)
            # print('Loss:', loss.item())
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()
        
        if args.test_run:
            epoch_time = time.time() - start_time
            print(f'Epoch: {epoch_time:.4f} seconds')
        epoch_loss = running_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                inputs = batch_inputs.double().to(device)
                labels = batch_labels.long().to(device)  # shape: (B, N, num_classes)
                B, N, F = inputs.shape
                inputs = inputs.permute(0, 2, 1)
                outputs = model(inputs)
                loss, acc = criterion(outputs, labels)
                val_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
        val_loss /= len(val_loader)
        val_acc = total_acc / total_batches
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        with open(os.path.join(run_dir, "log.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
        scheduler.step()

    # --- Testing Phase ---
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            inputs = batch_inputs.double().to(device)
            labels = batch_labels.long().to(device)
            B, N, F = inputs.shape
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs)
            loss, acc = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # import matplotlib.pyplot as plt

    # # Plot embeddings for five test samples
    # model.eval()
    # # Get one batch from the test loader
    # batch_inputs, batch_labels = next(iter(test_loader))
    # batch_inputs = batch_inputs.double().to(device)
    # batch_labels = batch_labels.long().to(device)
    # B, N, F = batch_inputs.shape
    # inputs_transposed = batch_inputs.permute(0, 2, 1)
    # embeddings = model(inputs_transposed)  # Shape: (B, N, projection_dim)
    # embeddings_np = embeddings.cpu().detach().numpy()
    # labels_np = batch_labels.cpu().detach().numpy()

    # for i in range(5):
    #     emb_sample = embeddings_np[i]      # (N, 2) expected
    #     label_sample = labels_np[i]          # (N,)
        
    #     plt.figure()
    #     scatter = plt.scatter(emb_sample[:, 0], emb_sample[:, 1], c=label_sample, cmap='viridis', s=10)
    #     plt.colorbar(scatter)
    #     plt.title(f"Contrastive Embedding - Sample {i+1}")
    #     plt.xlabel("Dimension 1")
    #     plt.ylabel("Dimension 2")
        
    #     plot_file = os.path.join(run_dir, f"embedding_sample_{i+1}.png")
    #     plt.savefig(plot_file)
    #     plt.close()
    # Save the final model state
    model_file = os.path.join(run_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Training complete. Model saved to {model_file}")

if __name__ == '__main__':
    main()