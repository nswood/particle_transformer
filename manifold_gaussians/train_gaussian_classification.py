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
    all_graph_dist_matrices = []
    for data_file in glob_paths:
        with h5py.File(data_file, 'r') as cur_file:
            cur_data = cur_file['datasets'][:]
            cur_labels = cur_file['labels'][:]
            cur_graph_dist_matrices = cur_file['graph_dist_matrices'][:]
            new_data = []
            new_labels = []
            new_data_graph_dist_matrices = []
            for i in range(cur_labels.shape[0]):
                sample_data = []
                sample_labels = []
                all_ids = []
                for lbl in [0, 1, 2, 3]:
                    idx = np.where(cur_labels[i] == lbl)[0][:15]
                    all_ids.append(idx)
                    sample_data.append(cur_data[i, idx, :])
                    sample_labels.append(cur_labels[i, idx])
                all_ids = np.concatenate(all_ids)
                graph_dist_matrix = cur_graph_dist_matrices[i][all_ids][:, all_ids]
                
                sample_data = np.concatenate(sample_data, axis=0)
                sample_labels = np.concatenate(sample_labels, axis=0)
                perm = np.random.permutation(sample_data.shape[0])
                sample_data = sample_data[perm]
                sample_labels = sample_labels[perm]
                sampled_graph_dist_matrix = graph_dist_matrix[perm][:, perm]
                new_data.append(sample_data)
                new_labels.append(sample_labels)
                new_data_graph_dist_matrices.append(sampled_graph_dist_matrix)
            cur_data = np.stack(new_data, axis=0)
            cur_labels = np.stack(new_labels, axis=0)
            all_data.append(cur_data)
            all_labels.append(cur_labels)
            all_graph_dist_matrices.append(new_data_graph_dist_matrices)
    data_array = np.concatenate(all_data)
    labels_array = np.concatenate(all_labels)
    all_graph_dist_matrices = np.concatenate(all_graph_dist_matrices)
    all_data = torch.tensor(data_array)
    all_labels = torch.tensor(labels_array)
    all_graph_dist_matrices = torch.tensor(all_graph_dist_matrices)
    all_labels = torch.nn.functional.one_hot(all_labels.long(), num_classes=4)
    return all_data, all_labels, all_graph_dist_matrices

def load_model(model_name, device, part_geom, part_dim, k, learnable, local_geom_weighting):
    # For the new dense model we set input_dim to 2 (as before) and use a fixed number of particles.
    input_dim = 2  

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
    
    # Determine the number of experts from the number of geometries.
    # (Since we assume a single geometry, this will be 1.)
    num_experts = len(curvature_list)

    
    # Create the DenseMoG_MLP model using these parameters.
    # Note: Even if you have only one expert, shared_expert is kept True so that the model
    # still follows the original mechanism.
    print(learnable)
    num_parts = 60
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
    
    
    proj_model = nn.Sequential(
        nn.Flatten(1),
        nn.Linear(num_parts * particle_dim * num_experts, num_parts*2),
        nn.LayerNorm(num_parts*2),
        nn.ReLU(),
        nn.Linear(num_parts*2, num_parts*4),
        nn.Unflatten(1, (num_parts, 4))
    )

    # Projection model remains similar.
    # Update to be larger to make more expressive
    # proj_model = nn.Sequential(
    #     nn.Linear(2 * particle_dim * num_experts, 8),
    #     nn.LayerNorm(8),
    #     nn.ReLU(),

    #     nn.Linear(8, 8),
    #     nn.LayerNorm(8),
    #     nn.ReLU(),

    #     nn.Linear(8, 4),
    #     nn.LayerNorm(4),
    #     nn.ReLU(),

    #     nn.Linear(4, 4)
    # )
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
    train_data, train_labels,train_graph_dist_matrices = load_data_from_globs(train_files)
    print(f'Loaded {len(train_data)} samples')

    print('Loading test dataset...')
    test_data, test_labels,test_graph_dist_matrices = load_data_from_globs(test_files)
    print(f'Loaded {len(test_data)} samples')

    print('Loading val dataset...')
    val_data, val_labels,val_graph_dist_matrices = load_data_from_globs(val_files)
    print(f'Loaded {len(val_data)} samples')

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels, train_graph_dist_matrices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels,test_graph_dist_matrices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels,val_graph_dist_matrices)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    base_loss = nn.CrossEntropyLoss(reduction='none')
    criterion = PermutationInvariantLossVectorized(base_loss, num_classes=4).to(device)
    if args.dist_reg:  
        d_reg = DistortionLoss(w_1 = 1)

    # --- Load Model ---
    # Note: The new load_model now returns a DenseMoG_MLP and a projection model.
    model, projection_model = load_model(args.model_name, device, args.part_geom, args.part_dim,
                                           args.part_curvature_init,
                                           args.part_curvature_trainable,
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
        cols = ["Epoch", "Train Loss", "Val Loss", "Val Accuracy"]
        cols += [f"Curvature {m.name} {i}" for i,m in enumerate(model.part_manifolds)]
        log_writer.writerow(cols)
    balance_weight = 0.00005
    # balance_weight = 0.0000
    poincare_geom = geoopt.PoincareBall(c = 1)
    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        projection_model.train()
        running_loss = 0.0
        train_acc_sum = 0
        if args.test_run:
            start_time = time.time()
        for batch_inputs, batch_labels, batch_dist_matrices in train_loader:
            inputs = batch_inputs.double().to(device)
            labels = batch_labels.long().to(device)
            # Original shape: (B, N, F). For DenseMoG_MLP, we need (C, N, B)
            # So permute with axes (2, 1, 0).
            inputs = inputs.permute(2, 1, 0)
            optimizer.zero_grad()
            proj_optimizer.zero_grad()
            # print('Inputs:', inputs.shape)
            # graph_dist_matrix = []
            # for i in range(inputs.shape[-1]):
                
            #     cur_inputs = inputs[:,:,i].permute(1,0)
            #     cur_dist_matrix = poincare_geom.dist_matrix(cur_inputs, cur_inputs)
            #     graph_dist_matrix.append(cur_dist_matrix)
            # graph_dist_matrix = torch.stack(graph_dist_matrix, dim=0)
            # print('Graph distance matrix:', graph_dist_matrix.shape)
            embed, local_geom_weights,proc_parts = model(inputs)         # DenseMoG_MLP expects (C, N, B) and returns dense features.
            # print('Embed:', embed.shape)
            # print('Proc_parts:', proc_parts.shape)
            outputs = projection_model(embed)
            loss, correct = criterion(outputs, labels)
            num_classes = outputs.size(1)
            train_acc_sum += correct
            predictions = torch.argmax(outputs, dim=1)
            balance_loss = class_balance_regularizer(predictions, num_classes)
            # print('Train Balance loss:', balance_weight * balance_loss)
            total_loss = loss + balance_weight * balance_loss
            if args.dist_reg:
                total_loss += d_reg(local_geom_weights, proc_parts,model.part_manifolds, batch_dist_matrices)
            total_loss.backward()

            optimizer.step()
            proj_optimizer.step()
            running_loss += total_loss.item()
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
            for batch_inputs, batch_labels, batch_dist_matrices in val_loader:
                inputs = batch_inputs.double().to(device)
                labels = batch_labels.long().to(device)
                inputs = inputs.permute(2, 1, 0)
                # graph_dist_matrix = poincare_geom.bdist(inputs, inputs)
                # print('Graph distance matrix:', graph_dist_matrix.shape)
                embed, local_geom_weights,proc_parts = model(inputs)
                outputs = projection_model(embed)
                loss, correct = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                balance_loss = class_balance_regularizer(predictions, num_classes)
                # print('Val Balance loss:', balance_weight * balance_loss)
                total_loss = loss + balance_weight * balance_loss
                if args.dist_reg:
                    total_loss += d_reg(local_geom_weights, proc_parts,model.part_manifolds, batch_dist_matrices)
                val_loss += total_loss.item()
                val_acc_sum += correct
                n_batches += 1
        val_loss /= len(val_loader)
        val_acc = val_acc_sum / n_batches
        train_acc = train_acc_sum / len(train_loader)
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
        for batch_inputs, batch_labels, batch_dist_matrices in test_loader:
            inputs = batch_inputs.double().to(device)
            labels = batch_labels.long().to(device)
            inputs = inputs.permute(2, 1, 0)
            embed, local_geom_weights,proc_parts = model(inputs)
            outputs = projection_model(embed)
            loss, correct = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            balance_loss = class_balance_regularizer(predictions, num_classes)
            total_loss = loss + balance_weight * balance_loss
            if args.dist_reg:
                    total_loss += d_reg(local_geom_weights, proc_parts,model.part_manifolds, batch_dist_matrices)
            test_loss += total_loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    model_file = os.path.join(run_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Training complete. Model saved to {model_file}")

    if args.visualize_outputs:
        import matplotlib.pyplot as plt
        _,_,best_preds, y_true_labels = criterion(outputs, labels, return_labels=True)

        for i in range(5):
            with torch.no_grad():
                cur_labels = best_preds[i].cpu().numpy()
                cur_true_labels = y_true_labels[i].cpu().numpy()
                cur_inputs = inputs[:,:,i]
                cur_inputs = cur_inputs.permute(1, 0)
                print('Cur_labels:',cur_labels.shape)
                print('Cur_true_labels:',cur_true_labels.shape)
                
                
                
                # Create a side-by-side plot for true labels and predicted labels.
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                # Extract coordinates and labels.
                x_coords = cur_inputs[:,0].cpu().numpy()
                y_coords = cur_inputs[:,1].cpu().numpy()
                
                # Left plot: True labels.
                sc0 = ax[0].scatter(x_coords, y_coords, c=cur_true_labels, cmap='viridis')
                ax[0].set_title(f"True Labels - Sample {i}")
                ax[0].set_xlabel("X")
                ax[0].set_ylabel("Y")
                fig.colorbar(sc0, ax=ax[0])
                
                # Right plot: Predicted labels.
                sc1 = ax[1].scatter(x_coords, y_coords, c=cur_labels, cmap='viridis')
                ax[1].set_title(f"Predicted Labels - Sample {i}")
                ax[1].set_xlabel("X")
                ax[1].set_ylabel("Y")
                fig.colorbar(sc1, ax=ax[1])
                
                plt.tight_layout()
                plt.show()
                plt.savefig(os.path.join(run_dir, f"sample_{i}_labels.png"))
        model.eval()

if __name__ == '__main__':
    main()
