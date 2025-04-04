import argparse
import os

import time
import csv
import h5py
import glob
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import PermutationInvariantTraining
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
# Append paths for custom modules
sys.path.append('/n/home11/nswood/weaver-core')
sys.path.append('/n/home11/nswood/MoG')

from PMSimCLR_loss import ManifoldSimCLRLoss
from weaver.nn.model.PMNN import PMNN
from weaver.nn.model.PM_utils import ManifoldNNLayer  # imported but not redefined here
import geoopt

# Import our new dense model from toy_utils (which now contains DenseMoG_MLP)
from manifold_gaussians.toy_utils import DenseMoG_MLP
from manifold_gaussians.losses import *

import math
import itertools

def load_data_from_globs(glob_paths):
    all_data = []
    all_labels = []
    for data_file in glob_paths:
        with h5py.File(data_file, 'r') as cur_file:
            cur_data = cur_file['datasets'][:]
            cur_labels = cur_file['labels'][:]
            all_data.append(cur_data)
            all_labels.append(cur_labels)
    data_array = np.concatenate(all_data)
    labels_array = np.concatenate(all_labels)
    all_data = torch.tensor(data_array)
    all_labels = torch.tensor(labels_array)
    all_labels = torch.nn.functional.one_hot(all_labels.long(), num_classes=6).float()
    return all_data, all_labels

def load_model(model_name, device, part_geom, part_dim, k, learnable, local_geom_weighting):
    # For the new dense model we set input_dim to 2 (as before) and use a fixed number of particles.
    input_dim = 3

    # Convert part_geom to a list. Since we assume no comma-separated inputs, it will have one element.
    geometries = [part_geom]
    
    # Convert part_dim to an integer.
    particle_dim = int(part_dim)
    
    # Convert curvature input. If k equals '-1', use an empty list; otherwise, convert it to float
    # and build a list whose length equals the number of geometries.
    
    
    k = k.replace("m", "-")
    
    if ',' in k:
        curvature_list = [float(x) for x in k.split(',')]
    else:
        curvature_list = [float(k)]
    
    # print('Curvature list:',curvature_list)
    
    # Determine the number of experts from the number of geometries.
    # (Since we assume a single geometry, this will be 1.)
    num_experts = len(curvature_list)

    
    # Create the DenseMoG_MLP model using these parameters.
    # Note: Even if you have only one expert, shared_expert is kept True so that the model
    # still follows the original mechanism.
    # print(learnable)
    num_parts = 200
    model = DenseMoG_MLP(
        input_dim=input_dim,
        n_parts=num_parts,
        local_geom_weighting=local_geom_weighting,
        local_geom_k_size=[5, 8, 12],
        part_experts=num_experts,
        part_expert_curvature_init=curvature_list,
        part_experts_dim=particle_dim,
        particle_feature_agg_method='add+norm',
        activation='relu',
        dropout_rate=0.0,
        learnable=learnable
    )
    
    
    # proj_model = nn.Sequential(
    #     nn.Flatten(1),
    #     nn.Linear(num_parts * particle_dim * num_experts, num_parts*2),
    #     nn.LayerNorm(num_parts*2),
    #     nn.ReLU(),
    #     nn.Linear(num_parts*2, num_parts*4),
    #     nn.Unflatten(1, (num_parts, 4))
    # )

        
    class TransformerBlock(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x):
            # x: [B, N, F]
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x

    class TransformerCloudClassifier(nn.Module):
        def __init__(self, in_features, dim=64, num_heads=4, num_layers=2, out_classes=4):
            super().__init__()
            self.input_proj = nn.Linear(in_features, dim)
            self.transformer = nn.Sequential(
                *[TransformerBlock(dim, num_heads) for _ in range(num_layers)]
            )
            # Global classification: classifier now operates on the pooled global representation.
            self.classifier = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, out_classes)
            )

        def forward(self, x):
            """
            x: [B, N, F]
            returns: [B, out_classes]
            """
            x = self.input_proj(x)   # [B, N, dim]
            x = self.transformer(x)  # [B, N, dim]
            # Aggregate features from all points into a single global feature.
            x = x.mean(dim=1)        # [B, dim]
            logits = self.classifier(x)  # [B, out_classes]
            return logits
    
    proj_model = TransformerCloudClassifier(in_features=particle_dim * num_experts, dim=16, num_heads=4, num_layers=2, out_classes=6)
    
    for m in proj_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model.to(device).double(), proj_model.to(device).double()


def main():
    parser = argparse.ArgumentParser(
        description="Training script for Gaussian clustering using manifold representations"
    )
    parser.add_argument('--data_dir', type=str,
                        default="/n/holystore01/LABS/iaifi_lab/Lab/nswood/testing_hyperbolic_gaussians_toy",
                        help="Path to directory containing 'train', 'test', and 'val' subdirectories with h5 files.")
    parser.add_argument('--outdir', type=str, default='testing_manifold_gaussians',
                        help="Directory to store training run outputs.")
    parser.add_argument('--model_name', type=str, default='model',
                        help="Model name to use (e.g., 'simple').")
    parser.add_argument('--part_geom', type=str, default='R',
                        help="Particle representation geometry")
    parser.add_argument('--part_dim', type=str, default='2',
                        help="Particle representation dimension")
    parser.add_argument('--part_curvature_init', type=str, default='m1',
                        help="Particle representation curvature initialization")
    parser.add_argument('--part_curvature_trainable', action='store_true',
                        help="Enable particle representation curvature training")
    parser.add_argument('--batch_size', type=int, default=25,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument('--test_run', action='store_true',
                        help="Run in test mode.")
    parser.add_argument('--local_geom_weighting', action='store_true',
                        help="Enable local geometry weighting."),
    parser.add_argument('--dist_reg', action='store_true',
                        help="Enable distortion regulator."),
    parser.add_argument('--visualize_outputs', action='store_true',
                        help="Visualize outputs.")
    args = parser.parse_args()

    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.outdir, f'run_{args.model_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_run:
        max_files = 25
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

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    if args.dist_reg:  
        d_reg = DistortionLoss(w_1 = 0.01)

    # --- Load Model ---
    # Note: The new load_model now returns a DenseMoG_MLP and a projection model.
    model, projection_model = load_model(args.model_name, device, args.part_geom, args.part_dim,
                                           args.part_curvature_init,
                                           args.part_curvature_trainable,
                                           args.local_geom_weighting)
    
    # Count parameters.
    num_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in projection_model.parameters())
    model_parameters = sum(p.numel() for p in model.parameters())
    proj_parameters = sum(p.numel() for p in projection_model.parameters())
    print("Model parameters count:", model_parameters)
    print("Projection model parameters count:", proj_parameters)
    
    params_file = os.path.join(run_dir, "model_parameters.csv")
    with open(params_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Number of model parameters", num_params])

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)
    proj_optimizer = torch.optim.Adam(projection_model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.001)
    scheduler_proj = optim.lr_scheduler.ReduceLROnPlateau(proj_optimizer, 'min', threshold=0.001)

    log_file = os.path.join(run_dir, "log.csv")
    with open(log_file, "w", newline="") as f:
        log_writer = csv.writer(f)
        cols = ["Epoch", "Train Loss", "Val Loss", "Val Accuracy"]
        cols += [f"Curvature {m.name} {i}" for i,m in enumerate(model.part_manifolds)]
        log_writer.writerow(cols)
  
    
    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        projection_model.train()
        running_loss = 0.0
        train_acc_sum = 0
        if args.test_run:
            start_time = time.time()
        for batch_inputs, batch_labels in train_loader:
            inputs = batch_inputs.double().to(device)
            
            labels = batch_labels.to(device)
            # Original shape: (B, N, F). For DenseMoG_MLP, we need (C, N, B)
            # So permute with axes (2, 1, 0).
            inputs = inputs.permute(2, 1, 0)
            optimizer.zero_grad()
            proj_optimizer.zero_grad()
            
            embed, local_geom_weights,proc_parts = model(inputs)         # DenseMoG_MLP expects (C, N, B) and returns dense features.
            
            outputs = projection_model(embed)
            loss = criterion(outputs, labels)
            correct = (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).float().sum()
            num_classes = outputs.size(1)
            train_acc_sum += correct
            predictions = torch.argmax(outputs, dim=1)
            loss.backward()

            optimizer.step()
            proj_optimizer.step()
            running_loss += loss.item()
            
        if args.test_run:
            epoch_time = time.time() - start_time
            print(f'Epoch: {epoch_time:.4f} seconds')
        
        epoch_loss = running_loss / len(train_loader)

        model.eval()
        projection_model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                inputs = batch_inputs.double().to(device)
                labels = batch_labels.to(device)
                
                inputs = inputs.permute(2, 1, 0)
                # print('Graph distance matrix:', graph_dist_matrix.shape)
                embed, local_geom_weights,proc_parts = model(inputs)
                outputs = projection_model(embed)
                loss = criterion(outputs, labels)
                correct = (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).float().sum()

                
                val_loss += loss.item()
                val_acc_sum += correct
                n_batches += 1
        val_loss /= len(val_loader)
        val_acc = val_acc_sum / len(val_dataset)
        train_acc = train_acc_sum / len(train_dataset)
        log_line = f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Train Acc: {train_acc:.4f}, Cur LR: {scheduler.get_last_lr()[0]:.6e}"

        cur_curvatures = [m.k.item() for m in model.part_manifolds]
        log_line += f", Curvatures: {cur_curvatures}"
        
        print(log_line)
        with open(os.path.join(run_dir, "log.txt"), "a") as log_f:
            log_f.write(log_line + "\n")
        # Prepare a CSV row that includes a column for each curvature value.
        csv_row = [epoch+1, f"{epoch_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"] + [f"{curv:.4f}" for curv in cur_curvatures]
        with open(log_file, "a", newline="") as f:
            log_writer = csv.writer(f)
            log_writer.writerow(csv_row)
        scheduler.step(val_loss)
        scheduler_proj.step(val_loss)

    model.eval()
    projection_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            inputs = batch_inputs.double().to(device)
            
            labels = batch_labels.to(device)
            inputs = inputs.permute(2, 1, 0)
            embed, local_geom_weights,proc_parts = model(inputs)
            outputs = projection_model(embed)
            loss = criterion(outputs, labels)
            correct = (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).float().sum()

            predictions = torch.argmax(outputs, dim=1)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    model_file = os.path.join(run_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Training complete. Model saved to {model_file}")


if __name__ == '__main__':
    main()
