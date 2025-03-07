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

import math
import itertools

class PermutationInvariantLossVectorized(nn.Module):
    def __init__(self, base_loss_fn, num_classes=4):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        perms = list(itertools.permutations(range(num_classes)))
        self.num_perms = len(perms)
        self.register_buffer('perm_tensor', torch.tensor(perms))

    def forward(self, y_pred, y_true):
        batch_size, num_points, num_classes = y_pred.shape
        # print('y_pred', y_pred.shape)
        # print('y_true', y_true.shape)
        if num_classes != self.perm_tensor.shape[1]:
            raise ValueError("Mismatch in number of classes between predictions and permutation tensor.")

        y_pred_expanded = y_pred.unsqueeze(1).expand(batch_size, self.num_perms, num_points, num_classes)
        perm_tensor_expanded = self.perm_tensor.view(1, self.num_perms, 1, num_classes).expand(batch_size, self.num_perms, num_points, num_classes)
        y_pred_permuted = torch.gather(y_pred_expanded, dim=3, index=perm_tensor_expanded)
        y_true_expanded = y_true.unsqueeze(1).expand(batch_size, self.num_perms, num_points, num_classes)

        losses = self.base_loss_fn(y_pred_permuted.permute(0, 3, 2, 1), y_true_expanded.permute(0, 3, 2, 1).float())
        losses = torch.sum(losses, dim=1).view(batch_size, self.num_perms)
        best_loss_values, best_perm_idx = losses.min(dim=1)
        loss_value = best_loss_values.mean()

        best_y_pred = y_pred_permuted[torch.arange(batch_size), best_perm_idx, :, :]
        best_preds = best_y_pred.argmax(dim=2)
        y_true_labels = y_true.argmax(dim=2)
        correct = (best_preds == y_true_labels).float().mean()

        return loss_value, correct

def load_data_from_globs(glob_paths):
    all_data = []
    all_labels = []
    for data_file in glob_paths:
        with h5py.File(data_file, 'r') as cur_file:
            cur_data = cur_file['datasets'][:]
            cur_labels = cur_file['labels'][:]
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
            all_data.append(cur_data)
            all_labels.append(cur_labels)
    data_array = np.concatenate(all_data)
    labels_array = np.concatenate(all_labels)
    all_data = torch.tensor(data_array)
    all_labels = torch.tensor(labels_array)
    all_labels = torch.nn.functional.one_hot(all_labels.long(), num_classes=4)
    return all_data, all_labels

def load_model(model_name, device, part_geom, part_dim, k, learnable, local_geom_weighting):
    # For the new dense model we set input_dim to 2 (as before) and use a fixed number of particles.
    input_dim = 2  

    # Convert part_geom to a list. Since we assume no comma-separated inputs, it will have one element.
    geometries = [part_geom]
    
    # Convert part_dim to an integer.
    particle_dim = int(part_dim)
    
    # Convert curvature input. If k equals '-1', use an empty list; otherwise, convert it to float
    # and build a list whose length equals the number of geometries.
    if ',' in k:
        curvature_list = [float(x) for x in k.split(',')]
    else:
        curvature_list = [float(k)]
    
    # Determine the number of experts from the number of geometries.
    # (Since we assume a single geometry, this will be 1.)
    num_experts = len(curvature_list)

    
    # Create the DenseMoG_MLP model using these parameters.
    # Note: Even if you have only one expert, shared_expert is kept True so that the model
    # still follows the original mechanism.
    model = DenseMoG_MLP(
        input_dim=input_dim,
        n_parts=400,
        local_geom_weighting=local_geom_weighting,
        local_geom_k_size=[5, 8, 12],
        part_experts=num_experts,
        part_expert_curvature_init=curvature_list,
        part_experts_dim=particle_dim,
        particle_feature_agg_method='add+norm',
        activation='relu',
        dropout_rate=0.1,
        learnable=learnable
    )
    
    # Projection model remains similar.
    
    proj_model = nn.Sequential(
        nn.Linear(2 * particle_dim, 4),
        nn.ReLU(),
        nn.Linear(4, 4)
    )
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
    parser.add_argument('--data_dir', type=str, default="/n/holystore01/LABS/iaifi_lab/Lab/nswood/testing_hyperbolic_gaussians_toy",
                        help="Path to directory containing 'train', 'test', and 'val' subdirectories with h5 files.")
    parser.add_argument('--outdir', type=str, default='testing_manifold_gaussians',
                        help="Directory to store training run outputs.")
    parser.add_argument('--model_name', type=str, default='model',
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
    parser.add_argument('--local_geom_weighting', type=bool, default=False,
                        help="Enable local geometry weighting.")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.outdir, f'run_{args.model_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_run:
        max_files = 5
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

    base_loss = nn.CrossEntropyLoss(reduction='none')
    criterion = PermutationInvariantLossVectorized(base_loss, num_classes=4).to(device)

    # --- Load Model ---
    # Note: The new load_model now returns a DenseMoG_MLP and a projection model.
    model, projection_model = load_model(args.model_name, device, args.part_geom, args.part_dim,
                                           args.part_curvature_init, args.part_curvature_trainable,
                                           args.local_geom_weighting)
    
    # Count parameters.
    num_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in projection_model.parameters())
    print("Model parameters count:", num_params)
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
        log_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        projection_model.train()
        running_loss = 0.0
        if args.test_run:
            start_time = time.time()
        for batch_inputs, batch_labels in train_loader:
            inputs = batch_inputs.double().to(device)
            labels = batch_labels.long().to(device)
            # Original shape: (B, N, F). For DenseMoG_MLP, we need (C, N, B)
            # So permute with axes (2, 1, 0).
            inputs = inputs.permute(2, 1, 0)
            optimizer.zero_grad()
            proj_optimizer.zero_grad()

            embed = model(inputs)         # DenseMoG_MLP expects (C, N, B) and returns dense features.
            # print('embed', torch.isnan(embed).any())
            # print('embed', embed.shape)
            outputs = projection_model(embed)
            # print('outputs', outputs.shape)
            # print('labels', labels.shape)
            # print('max embed', torch.max(embed))
            # print('min embed', torch.min(embed))
            # print('outputs', torch.isnan(outputs).any())
            # print('labels', torch.isnan(labels).any())
            loss, correct = criterion(outputs, labels)
            # print('loss', torch.isnan(loss).any())
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
                labels = batch_labels.long().to(device)
                inputs = inputs.permute(2, 1, 0)
                embed = model(inputs)
                outputs = projection_model(embed)
                loss, correct = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc_sum += correct
                n_batches += 1
        val_loss /= len(val_loader)
        val_acc = val_acc_sum / n_batches
        log_line = f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Cur LR: {scheduler.get_last_lr()[0]:.6e}"
        print(log_line)
        with open(os.path.join(run_dir, "log.txt"), "a") as log_f:
            log_f.write(log_line + "\n")
        with open(log_file, "a", newline="") as f:
            log_writer = csv.writer(f)
            log_writer.writerow([epoch+1, f"{epoch_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])
        scheduler.step(val_loss)
        scheduler_proj.step(val_loss)

    model.eval()
    projection_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            inputs = batch_inputs.double().to(device)
            labels = batch_labels.long().to(device)
            inputs = inputs.permute(2, 1, 0)
            embed = model(inputs)
            outputs = projection_model(embed)
            loss, correct = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    model_file = os.path.join(run_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Training complete. Model saved to {model_file}")

if __name__ == '__main__':
    main()


# import argparse
# import os
# import sys
# import time
# import csv
# import h5py
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchmetrics import PermutationInvariantTraining
# import glob
# import numpy as np
# import itertools
# from PMSimCLR_loss import ManifoldSimCLRLoss
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# sys.path.append('/n/home11/nswood/weaver-core')
# sys.path.append('/n/home11/nswood/MoG')
# from weaver.nn.model.PMNN import PMNN
# from weaver.nn.model.PM_utils import ManifoldNNLayer
# import geoopt
# from manifold_gaussians.toy_utils import *

# class PermutationInvariantLossVectorized(nn.Module):
#     def __init__(self, base_loss_fn, num_classes=4):
#         """
#         Args:
#             base_loss_fn (callable): A function that takes (pred, target) and returns a tensor
#                                      of shape (batch_size*num_perms,) containing per-sample losses.
#                                      The inputs should have shape (batch_size*num_perms, num_points, num_classes).
#             num_classes (int): Number of classes.
#         """
#         super(PermutationInvariantLossVectorized, self).__init__()
#         self.base_loss_fn = base_loss_fn
#         perms = list(itertools.permutations(range(num_classes)))
#         self.num_perms = len(perms)
#         # Register the permutation tensor as a buffer so it moves with the model.
#         self.register_buffer('perm_tensor', torch.tensor(perms))

#     def forward(self, y_pred, y_true):
#         """
#         Args:
#             y_pred (torch.Tensor): Predictions with shape (batch_size, num_points, num_classes)
#             y_true (torch.Tensor): Ground truth with shape (batch_size, num_points, num_classes)
#         Returns:
#             tuple: (averaged loss over the batch after selecting the best permutation, accuracy)
#         """
#         batch_size, num_points, num_classes = y_pred.shape
#         if num_classes != self.perm_tensor.shape[1]:
#             raise ValueError("Mismatch in number of classes between predictions and permutation tensor.")

#         # Expand y_pred to shape: (batch_size, num_perms, num_points, num_classes)
#         y_pred_expanded = y_pred.unsqueeze(1).expand(batch_size, self.num_perms, num_points, num_classes)
#         # Expand the permutation tensor: shape (batch_size, num_perms, num_points, num_classes)
#         perm_tensor_expanded = self.perm_tensor.view(1, self.num_perms, 1, num_classes).expand(batch_size, self.num_perms, num_points, num_classes)
#         # Permute the class dimension for all samples.
#         y_pred_permuted = torch.gather(y_pred_expanded, dim=3, index=perm_tensor_expanded)
#         # Expand y_true to shape: (batch_size, num_perms, num_points, num_classes)
#         y_true_expanded = y_true.unsqueeze(1).expand(batch_size, self.num_perms, num_points, num_classes)

#         # Compute loss per sample per permutation.
#         losses = self.base_loss_fn(y_pred_permuted.permute(0, 3, 2, 1), y_true_expanded.permute(0, 3, 2, 1).float())
#         losses = torch.sum(losses, dim=1)
#         losses = losses.view(batch_size, self.num_perms)

#         # For each sample, find the permutation with the minimal loss.
#         best_loss_values, best_perm_idx = losses.min(dim=1)
#         loss_value = best_loss_values.mean()

#         # Compute accuracy using predictions corresponding to the best permutation.
#         # Get the best predictions: shape (batch_size, num_points, num_classes)
#         best_y_pred = y_pred_permuted[torch.arange(batch_size), best_perm_idx, :, :]
#         best_preds = best_y_pred.argmax(dim=2)
#         y_true_labels = y_true.argmax(dim=2)
#         # Calculate overall accuracy across the batch.
#         correct = (best_preds == y_true_labels).float().mean()

#         return loss_value, correct

# def load_data_from_globs(glob_paths):
#         all_data = []
#         all_labels = []
#         for data_file in glob_paths:
#             # print(f'Loading {data_file}...')
#             with h5py.File(data_file, 'r') as cur_file:
#                 # Directly convert the dataset to torch tensor to avoid extra numpy concatenation
#                 cur_data = cur_file['datasets'][:]
#                 cur_labels = cur_file['labels'][:]
#                 # print('Cur Data:', cur_data.shape)
#                 # print('Cur Labels:', cur_labels.shape)
#                 new_data = []
#                 new_labels = []
#                 for i in range(cur_labels.shape[0]):
#                     sample_data = []
#                     sample_labels = []
#                     for lbl in [0, 1, 2, 3]:
#                         idx = np.where(cur_labels[i] == lbl)[0][:100]
#                         sample_data.append(cur_data[i, idx, :])
#                         sample_labels.append(cur_labels[i, idx])
#                     sample_data = np.concatenate(sample_data, axis=0)
#                     sample_labels = np.concatenate(sample_labels, axis=0)
                    
#                     perm = np.random.permutation(sample_data.shape[0])
#                     sample_data = sample_data[perm]
#                     sample_labels = sample_labels[perm]

#                     new_data.append(sample_data)
#                     new_labels.append(sample_labels)
#                 cur_data = np.stack(new_data, axis=0)
#                 cur_labels = np.stack(new_labels, axis=0)
#                 # print('Cur Data:', cur_data.shape)
#                 # print('Cur Labels:', cur_labels.shape)
#                 # print("Label counts:", dict(zip(unique_labels, label_counts)))
#                 all_data.append(cur_data)
#                 all_labels.append(cur_labels)

#         data_array = np.concatenate(all_data)
#         labels_array = np.concatenate(all_labels)
#         all_data = torch.tensor(data_array)
#         all_labels = torch.tensor(labels_array)
#         all_labels = torch.nn.functional.one_hot(all_labels.long(), num_classes=4)
        
#         return all_data, all_labels 
  

# def load_model(model_name, device, part_geom, part_dim, k, learnable):
    
#     input_dim = 2
#     output_dim = 4
#     if ',' in part_geom:
#         n_manifolds = len(part_geom.split(','))
#     else:
#         n_manifolds = 1
    
#     model = ParticleCLRModel(input_dim, output_dim, part_geom, part_dim, k=k, learnable=learnable, model_name=model_name)
    
#     input_dim = input_dim * n_manifolds
#     proj_model = nn.Sequential(
#         nn.Linear(2 * input_dim, output_dim),
#         nn.ReLU(),
#         nn.LayerNorm(output_dim),
#         nn.Linear(output_dim, output_dim)
#     )
#     return model.to(device).double(), proj_model.double()

# def main():
#     parser = argparse.ArgumentParser(
#         description="Training script for Gaussian clustering using manifold representations"
#     )
#     parser.add_argument('--data_dir', type=str, 
#                         help="Path to directory containing 'train', 'test', and 'val' subdirectories with h5 files.", default = "/n/holystore01/LABS/iaifi_lab/Lab/nswood/testing_hyperbolic_gaussians_toy")
#     parser.add_argument('--outdir', type=str,
#                         help="Directory to store training run outputs.", default = 'testing_manifold_gaussians')
#     parser.add_argument('--model_name', type=str, default = 'model',
#                         help="Model name to use (e.g., 'simple').")
#     parser.add_argument('--part_geom', type=str, default='R',
#                         help="Particle representation geometry")
#     parser.add_argument('--part_dim', type=str, default='2',
#                         help="Particle representation dimension")
#     parser.add_argument('--part_curvature_init', type=str, default='-1',
#                         help="Particle representation curvature initialization")
#     parser.add_argument('--part_curvature_trainable', type=bool, default=True,
#                         help="Particle representation curvature trainable")
#     parser.add_argument('--batch_size', type=int, default=25,
#                         help="Batch size for training.")
#     parser.add_argument('--lr', type=float, default=0.001,
#                         help="Learning rate.")
#     parser.add_argument('--epochs', type=int, default=10,
#                         help="Number of training epochs.")
#     parser.add_argument('--test_run', type=bool, default=False,
#                         help="test_run.")
#     parser.add_argument('--local_geom_weighting', type=bool, default=False,
#                         help="test_run.")
#     args = parser.parse_args()

#     # Create a new run directory under the provided outdir
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     run_dir = os.path.join(args.outdir, f'run_{args.model_name}_{timestamp}')
#     os.makedirs(run_dir, exist_ok=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load dataset from an h5 file
    

#     # Load the dataset from the provided data directory
#     if args.test_run:
#         max_files = 5
#     else:
#         max_files = np.inf

#     train_pattern = os.path.join(args.data_dir, 'train', '*.h5')
#     train_files = glob.glob(train_pattern)
#     test_pattern = os.path.join(args.data_dir, 'test', '*.h5')
#     test_files = glob.glob(test_pattern)
#     val_pattern = os.path.join(args.data_dir, 'val', '*.h5')
#     val_files = glob.glob(val_pattern)

#     if len(train_files) > max_files:
#         train_files = train_files[:max_files]
#     if len(test_files) > max_files:
#         test_files = test_files[:max_files]
#     if len(val_files) > max_files:
#         val_files = val_files[:max_files]

#     print('Loading train dataset...')
#     train_data, train_labels = load_data_from_globs(train_files)
#     print(f'Loaded {len(train_data)} samples')


#     print('Loading test dataset...')
#     test_data, test_labels = load_data_from_globs(test_files)
#     print(f'Loaded {len(test_data)} samples')

    
#     print('Loading val dataset...')
#     val_data, val_labels = load_data_from_globs(val_files)
#     print(f'Loaded {len(val_data)} samples')

            
#     # Create data loaders
#     train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

#     test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#     val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

#         # --- Supervised Contrastive Loss (CLR) ---
#     base_loss = nn.CrossEntropyLoss(reduction='none')
#     criterion = PermutationInvariantLossVectorized(base_loss, num_classes=4).to(device)

    

#     # --- Training Setup ---
#     # Example: load the model (this function should return a ParticleCLRModel instance)
#     model, projection_model = load_model(args.model_name, device, args.part_geom, args.part_dim, args.part_curvature_init, args.part_curvature_trainable, args.local_geom_weighting)
    
#     model = model.to(device)
#     projection_model = projection_model.to(device) 



#     # Count model parameters and write them to a CSV file.
#     num_params = sum(p.numel() for p in model.parameters())
#     num_params += sum(p.numel() for p in projection_model.parameters())
#     print("Model parameters count:", num_params)
#     params_file = os.path.join(run_dir, "model_parameters.csv")
#     with open(params_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Metric", "Value"])
#         writer.writerow(["Number of model parameters", num_params])

#     # Create optimizers for both models.
#     optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)
#     proj_optimizer = torch.optim.Adam(projection_model.parameters(), lr=args.lr)

#     # Create learning rate schedulers.
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', threshold = 0.001)
#     scheduler_proj = optim.lr_scheduler.ReduceLROnPlateau(proj_optimizer, 'min', threshold = 0.001)

#     # Prepare the CSV log file and write header.
#     log_file = os.path.join(run_dir, "log.csv")
#     with open(log_file, "w", newline="") as f:
#         log_writer = csv.writer(f)
#         log_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

#     # --- Training Loop ---
#     for epoch in range(args.epochs):
#         model.train()
#         projection_model.train()  # Ensure projection_model is in train mode as well.
#         running_loss = 0.0

#         if args.test_run:
#             start_time = time.time()
#         for batch_inputs, batch_labels in train_loader:
#             # Ensure inputs are floats and labels are longs
#             inputs = batch_inputs.double().to(device)
#             labels = batch_labels.long().to(device)  # shape: (B, N)

#             # Prepare inputs: shape (B, N, F) -> (B, F, N)
#             B, N, F = inputs.shape
#             inputs = inputs.permute(0, 2, 1)

#             # Zero gradients for both optimizers
#             optimizer.zero_grad()
#             proj_optimizer.zero_grad()

#             # Forward pass through embedding model then projection model
#             embed = model(inputs)                     # embed shape: (B, N, projection_dim)
#             outputs = projection_model(embed)         # outputs shape: (B, N, projection_dim) or similar

#             # Compute supervised contrastive loss using labels
#             loss, correct = criterion(outputs, labels)
#             loss.backward()

#             # Update weights for both models
#             optimizer.step()
#             proj_optimizer.step()

#             running_loss += loss.item()
        
#         if args.test_run:
#             epoch_time = time.time() - start_time
#             print(f'Epoch: {epoch_time:.4f} seconds')
#         epoch_loss = running_loss / len(train_loader)
        
#         # --- Validation Phase ---
#         model.eval()
#         projection_model.eval()
#         val_loss = 0.0
#         val_acc_sum = 0.0
#         n_batches = 0
#         with torch.no_grad():
#             for batch_inputs, batch_labels in val_loader:
#                 inputs = batch_inputs.double().to(device)
#                 labels = batch_labels.long().to(device)  # shape: (B, N)
#                 B, N, F = inputs.shape
#                 inputs = inputs.permute(0, 2, 1)
#                 embed = model(inputs)
#                 outputs = projection_model(embed)
#                 loss, correct = criterion(outputs, labels)
#                 val_acc_sum += correct
#                 n_batches += 1
#                 val_loss += loss.item()
#         val_loss /= len(val_loader)
#         val_acc = val_acc_sum / n_batches
#         log_line = f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Cur LR: {scheduler.get_last_lr()[0]:.6e}"
#         print(log_line)
#         with open(os.path.join(run_dir, "log.txt"), "a") as log_f:
#             log_f.write(log_line + "\n")
        
#         # Append current epoch results to the CSV log.
#         with open(log_file, "a", newline="") as f:
#             log_writer = csv.writer(f)
#             log_writer.writerow([epoch+1, f"{epoch_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

#         # Step the learning rate schedulers for both optimizers
#         scheduler.step(val_loss)
#         scheduler_proj.step(val_loss)

#     # --- Testing Phase ---
#     model.eval()
#     projection_model.eval()
#     test_loss = 0.0
#     with torch.no_grad():
#         for batch_inputs, batch_labels in test_loader:
#             inputs = batch_inputs.double().to(device)
#             labels = batch_labels.long().to(device)
#             B, N, F = inputs.shape
#             inputs = inputs.permute(0, 2, 1)
#             embed = model(inputs)
#             outputs = projection_model(embed)
#             loss, correct = criterion(outputs, labels)
#             test_loss += loss.item()
#     test_loss /= len(test_loader)
#     print(f"Test Loss: {test_loss:.4f}")

#     # Save the final model state.
#     model_file = os.path.join(run_dir, f"{args.model_name}_final.pth")
#     torch.save(model.state_dict(), model_file)
#     print(f"Training complete. Model saved to {model_file}")


# if __name__ == '__main__':
#     main()