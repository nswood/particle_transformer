import argparse
import os
import time
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import PermutationInvariantTraining
import glob
#!/usr/bin/env python3

import torch.nn as nn
import torch.optim as optim

def load_data_from_globs(glob_paths):
        all_data = []
        all_labels = []
        for data_file in glob_paths:
            with h5py.File(data_file, 'r') as cur_file:
                # Directly convert the dataset to torch tensor to avoid extra numpy concatenation
                all_data.append(torch.tensor(cur_file['datasets'][:]))
                all_labels.append(torch.tensor(cur_file['labels'][:]))
        
        all_data = torch.concat(all_data, dim=0)
        all_labels = torch.concat(all_labels, dim=0)
        return all_data, all_labels 
# Skeleton function for loading a model.
def load_model(model_name, device):
    if model_name == 'simple':
        # Example simple model
        model = nn.Sequential(
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 1600)
        )
    else:
        # Default model
        model = nn.Sequential(
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 1600)
        )
    return model.to(device).double()

def main():
    parser = argparse.ArgumentParser(
        description="Training script for Gaussian clustering using manifold representations"
    )
    parser.add_argument('--data_dir', type=str, 
                        help="Path to directory containing 'train', 'test', and 'val' subdirectories with h5 files.", default = "/n/holystore01/LABS/iaifi_lab/Lab/nswood/testing_hyperbolic_gaussians_toy")
    parser.add_argument('--outdir', type=str,
                        help="Directory to store training run outputs.", default = 'tesing_manifold_gaussians')
    parser.add_argument('--model_name', type=str, default='test',
                        help="Model name to use (e.g., 'simple').")
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
    model = load_model(args.model_name, device)

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
            print('Inputs type:', inputs.dtype)
            print('Labels type:', labels.dtype)
            num_classes = int(labels.max().item() + 1)
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
            B, N, F = inputs.shape

            inputs = inputs.view(inputs.size(0), -1)
            
            print(
                f"Inputs shape: {inputs.shape}, labels shape: {labels.shape}"
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), N, -1)
            outputs = torch.nn.Softmax(dim=-1)(outputs).to(device)
            print('Outputs shape:', outputs.shape)
            print('Outputs type:', outputs.dtype)
            loss = criterion(outputs, labels)
            print('Loss:', loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.double().to(device)
                labels = labels.long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
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