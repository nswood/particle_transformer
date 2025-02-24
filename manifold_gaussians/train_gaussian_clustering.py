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
#!/usr/bin/env python3

import torch.nn as nn
import torch.optim as optim
sys.path.append('/n/home11/nswood/weaver-core')
from weaver.nn.model.PMNN import PMNN

class cluster_classifier_model(nn.Module):
    def __init__(self, input_dim, output_dim, embedder_model):
        super(cluster_classifier_model, self).__init__()
        self.embedder_model = embedder_model
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 800),
            nn.ReLU(),
            nn.Linear(800, output_dim)
        )
    def forward(self, x):
        B,N,F = x.shape
        tan_x = self.embedder_model(x)
        if type(tan_x) == tuple:
            tan_x = tan_x[1]
        tan_x = tan_x.view(tan_x.size(0), -1)
        x = self.classifier(tan_x)
        x = x.view(B, N, -1)
        return x


def load_data_from_globs(glob_paths):
        all_data = []
        all_labels = []
        for data_file in glob_paths:
            with h5py.File(data_file, 'r') as cur_file:
                # Directly convert the dataset to torch tensor to avoid extra numpy concatenation
                all_data.append(cur_file['datasets'][:])
                all_labels.append(cur_file['labels'][:])

        data_array = np.concatenate(all_data)
        labels_array = np.concatenate(all_labels)
        all_data = torch.tensor(data_array)
        all_labels = torch.tensor(labels_array)
        return all_data, all_labels 

# Skeleton function for loading a model.
def load_model(model_name, device, part_geom, part_dim, part_curvature_init, part_curvature_trainable):
    if model_name == 'test':
        if part_dim is None:
            part_dim = 2
        # Example simple model
        embedder = nn.Sequential(
            nn.Linear(2, part_dim),
            nn.ReLU(),
            nn.Linear(part_dim, part_dim)
        )
        
    else:
        print('Building PM-MLP model')
        print('part_geom:', part_geom)
        print('part_dim:', part_dim)
        print('part_curvature_init:', part_curvature_init)
        print('part_curvature_trainable:', part_curvature_trainable)
        # Default model
        embedder = PMNN(2,
            part_geom = part_geom,
            part_dim =  part_dim,
            part_curvature_init = part_curvature_init,
            learnable = part_curvature_trainable)
    flatten_input = int(part_dim) * 400
    flatten_output = 4 * 400
    model = cluster_classifier_model(flatten_input, flatten_output, embedder)
    return model.to(device).double()

def main():
    parser = argparse.ArgumentParser(
        description="Training script for Gaussian clustering using manifold representations"
    )
    parser.add_argument('--data_dir', type=str, 
                        help="Path to directory containing 'train', 'test', and 'val' subdirectories with h5 files.", default = "/n/holystore01/LABS/iaifi_lab/Lab/nswood/testing_hyperbolic_gaussians_toy")
    parser.add_argument('--outdir', type=str,
                        help="Directory to store training run outputs.", default = 'tesing_manifold_gaussians')
    parser.add_argument('--model_name', type=str,
                        help="Model name to use (e.g., 'simple').")
    parser.add_argument('--part_geom', type=str, default='R',
                        help="Particle representation geometry")
    parser.add_argument('--part_dim', type=str, default='2',
                        help="Particle representation dimension")
    parser.add_argument('--part_curvature_init', type=str, default='-1',
                        help="Particle representation curvature initialization")
    parser.add_argument('--part_curvature_trainable', type=bool, default=True,
                        help="Particle representation curvature trainable")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs.")
    args = parser.parse_args()

    # Create a new run directory under the provided outdir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.outdir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset from an h5 file
    

    # Load the dataset from the provided data directory
    train_pattern = os.path.join(args.data_dir, 'train', '*.h5')
    train_files = glob.glob(train_pattern)
    print('Loading train dataset...')
    train_data, train_labels = load_data_from_globs(train_files)
    print(f'Loaded {len(train_data)} samples')


    test_pattern = os.path.join(args.data_dir, 'test', '*.h5')
    test_files = glob.glob(test_pattern)
    print('Loading test dataset...')
    test_data, test_labels = load_data_from_globs(test_files)
    print(f'Loaded {len(test_data)} samples')

    val_pattern = os.path.join(args.data_dir, 'val', '*.h5')
    val_files = glob.glob(val_pattern)
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

    # Load the model based on model_name
    model = load_model(args.model_name, device, args.part_geom, args.part_dim, args.part_curvature_init, args.part_curvature_trainable)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = PermutationInvariantTraining(metric_func=criterion, eval_func = 'min').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # Training loop using train, val, and test dataloaders
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            # Ensure inputs are floats and labels are longs (for classification)
            inputs = batch_inputs.double().to(device)
            labels = batch_labels.long().to(device)
            num_classes = int(labels.max().item() + 1)
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
            B, N, F = inputs.shape
            inputs = inputs.permute(0, 2, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), N, -1)
            loss = criterion(outputs, labels)
            print('Train Loss:', loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Validation phase updated to follow training setup
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                inputs = batch_inputs.double().to(device)
                labels = batch_labels.long().to(device)
                num_classes = int(labels.max().item() + 1)
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
                B, N, F = inputs.shape
                inputs = inputs.permute(0, 2, 1)

                outputs = model(inputs)
                outputs = outputs.view(outputs.size(0), N, -1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                print('Val Loss:', loss.item())

                # Compute predictions and accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=2)).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = (correct / total) * 100

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        with open(os.path.join(run_dir, "log.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")

        if scheduler is not None:
            scheduler.step()


    # Testing phase after training
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.double().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    test_accuracy = (correct / total) * 100
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save the final model state
    model_file = os.path.join(run_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Training complete. Model saved to {model_file}")

if __name__ == '__main__':
    main()