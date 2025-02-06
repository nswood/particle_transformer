import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# Step 1. Data Loading Functions
# -------------------------------
def load_data_from_folder(folder, use_additional_features=True):
    """
    Loads all CSV files in a folder, filters out rows with infinite or NaN values,
    and returns the combined features and labels.
    
    Parameters:
      folder (str): Directory containing CSV files.
      use_additional_features (bool): If True, include the extra features (delta, rel_delta, c);
                                      otherwise, use only the basic features.
    
    Returns:
      X_all (np.ndarray): Feature array (float32).
      y_all (np.ndarray): Label array (as strings).
    """
    file_paths = glob.glob(os.path.join(folder, "*.csv"))
    data_list = []
    labels_list = []
    
    for file in file_paths:
        # print(f"Loading data from {file}...")
        # Load CSV data
        df = pd.read_csv(file)
        # Drop the "index" column if it exists
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        
        # print(f"  Original shape: {df.shape}")
        # Replace infinite values with NaN, then drop any row with NaN values
        # df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # df.dropna(inplace=True)
        
        # If the dataframe is empty after dropping NaNs, skip this file.
        if df.empty:
            print('  No valid data in the file. Skipping...')
            continue
        
        # Extract the label from the filename.
        # For example: "ClassLabel_100_gromov_delta_results.csv" -> "ClassLabel"
        base_name = os.path.basename(file)
        label = base_name.split("_")[0]
        print(f"  Label: {label}")
        
        # Select features based on the flag
        if use_additional_features:
            # Use all features (10 basic features + 3 additional)
            feature_cols = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                            'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4',
                            'delta', 'rel_delta', 'c']
        else:
            # Use only the basic 10 features
            feature_cols = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                            'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4']
            
        # Ensure the dataframe has the required columns
        df = df[feature_cols]
        X = df.values.astype(np.float32)  # ensure data is float32
        
        # Create an array of labels (one label for each row in the CSV)
        y = np.full((X.shape[0],), label)
        
        
        data_list.append(X)
        labels_list.append(y)
    
    # Concatenate all data and labels along the row axis
    if data_list:
        X_all = np.concatenate(data_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0)
    else:
        X_all, y_all = np.array([]), np.array([])
    return X_all, y_all

# -------------------------------
# Step 2. Load Data from Directories
# -------------------------------
# Updated paths
train_dir = "JetClass_gromov_delta_results_EMD/train"
val_dir   = "JetClass_gromov_delta_results_EMD/val"
test_dir  = "JetClass_gromov_delta_results_EMD/test"

# Experiment A: Basic features (without delta, rel_delta, c)
X_train_simple, y_train_simple = load_data_from_folder(train_dir, use_additional_features=False)
X_val_simple,   y_val_simple   = load_data_from_folder(val_dir,   use_additional_features=False)
X_test_simple,  y_test_simple  = load_data_from_folder(test_dir,  use_additional_features=False)

# Experiment B: Full features (with delta, rel_delta, c)
X_train_full, y_train_full = load_data_from_folder(train_dir, use_additional_features=True)
X_val_full,   y_val_full   = load_data_from_folder(val_dir,   use_additional_features=True)
X_test_full,  y_test_full  = load_data_from_folder(test_dir,  use_additional_features=True)

# -------------------------------
# Step 3. Encode Class Labels
# -------------------------------
# We'll use the labels from the basic features experiment for consistency.
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_simple)
y_val_enc   = le.transform(y_val_simple)
y_test_enc  = le.transform(y_test_simple)
num_classes = len(le.classes_)

# -------------------------------
# Step 4. Create PyTorch Datasets and DataLoaders
# -------------------------------
batch_size = 32

def create_dataloader(X, y, batch_size, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# DataLoaders for simple (basic) features
train_loader_simple = create_dataloader(X_train_simple, y_train_enc, batch_size=batch_size, shuffle=True)
val_loader_simple   = create_dataloader(X_val_simple,   y_val_enc,   batch_size=batch_size, shuffle=False)
test_loader_simple  = create_dataloader(X_test_simple,  y_test_enc,  batch_size=batch_size, shuffle=False)

# DataLoaders for full features
train_loader_full = create_dataloader(X_train_full, y_train_enc, batch_size=batch_size, shuffle=True)
val_loader_full   = create_dataloader(X_val_full,   y_val_enc,   batch_size=batch_size, shuffle=False)
test_loader_full  = create_dataloader(X_test_full,  y_test_enc,  batch_size=batch_size, shuffle=False)

# -------------------------------
# Step 5. Define the MLP Model with Batch Norm on the Input
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            # Batch normalization on input features
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Step 6. Training and Evaluation Functions
# -------------------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Create logging directories
log_dir = "high_level_discriminator_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
run_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir)

# CSV file paths
csv_file_simple = os.path.join(run_dir, "performance_simple.csv")
csv_file_full = os.path.join(run_dir, "performance_full.csv")

# Initialize CSV files
with open(csv_file_simple, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

with open(csv_file_full, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

# -------------------------------
# Step 7. Training the Models
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20

# Create two models: one for basic features and one for full features.
model_simple = MLP(input_dim=X_train_simple.shape[1], num_classes=num_classes).to(device)
model_full   = MLP(input_dim=X_train_full.shape[1], num_classes=num_classes).to(device)

# Define loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer_simple = optim.Adam(model_simple.parameters(), lr=0.001)
optimizer_full   = optim.Adam(model_full.parameters(), lr=0.001)

print("Training model with basic features (without delta, rel_delta, c)...")
for epoch in range(num_epochs):
    train_loss_simple = train_model(model_simple, train_loader_simple, criterion, optimizer_simple, device)
    val_acc_simple = evaluate_model(model_simple, val_loader_simple, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss_simple:.4f} - Val Acc: {val_acc_simple:.4f}")

    # Log performance for simple model
    with open(csv_file_simple, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss_simple, None, None, val_acc_simple])

print("\nTraining model with full features (including delta, rel_delta, c)...")
for epoch in range(num_epochs):
    train_loss_full = train_model(model_full, train_loader_full, criterion, optimizer_full, device)
    val_acc_full = evaluate_model(model_full, val_loader_full, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss_full:.4f} - Val Acc: {val_acc_full:.4f}")

    # Log performance for full model
    with open(csv_file_full, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss_full, None, None, val_acc_full])

# -------------------------------
# Step 8. Evaluate on the Test Set
# -------------------------------
test_acc_simple = evaluate_model(model_simple, test_loader_simple, device)
print(f"\nTest Accuracy (Basic Features): {test_acc_simple:.4f}")

test_acc_full = evaluate_model(model_full, test_loader_full, device)
print(f"Test Accuracy (Full Features): {test_acc_full:.4f}")
