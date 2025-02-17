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
import torch.optim.lr_scheduler as lr_scheduler

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
        # print(f"  Label: {label}")
        # Select features based on the flag
        if use_additional_features:
            # Use all features (10 basic features + 3 additional)
            feature_cols = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                            'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4',
                            'delta', 'rel_delta']
        else:
            # Use only the basic 10 features
            feature_cols = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                            'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4']
            
        # Ensure the dataframe has the required columns
        df = df[feature_cols]
        X = df.values.astype(np.float32)  # ensure data is float32
        y = np.full((X.shape[0],), label)
        # print(X[0:5])
        nan_mask = np.isnan(X).any(axis=1)
        inf_mask = np.isinf(X).any(axis=1)
        invalid_mask = nan_mask | inf_mask
        if invalid_mask.any():
            X = X[~invalid_mask]
            y = y[~invalid_mask]
            print(f"  Removed {invalid_mask.sum()} rows with NaN or infinite values.")
            print(f"  Processed shape: {X.shape}")
            print('features', np.isnan(X).any() or np.isinf(X).any())
        
        # Create an array of labels (one label for each row in the CSV)
        
        
        # print('Check nan', np.isnan(X).any())
        data_list.append(X)
        labels_list.append(y)
    
    # Concatenate all data and labels along the row axis
    if data_list:
        X_all = np.concatenate(data_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0)
    else:
        X_all, y_all = np.array([]), np.array([])

    
    return X_all, y_all

# Create logging directories
log_dir = "high_level_discriminator_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
run_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir)

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
# Encode class labels and save the mapping to the log directory
y_train_enc_simple = le.fit_transform(y_train_simple)
y_val_enc_simple   = le.transform(y_val_simple)

# Save the label mapping to a CSV file
label_mapping_file = os.path.join(run_dir, "label_mapping.csv")
label_mapping = pd.DataFrame({'class': le.classes_, 'encoded_label': range(len(le.classes_))})
label_mapping.to_csv(label_mapping_file, index=False)
y_test_enc_simple  = le.transform(y_test_simple)

y_train_enc_full = le.transform(y_train_full)
y_val_enc_full   = le.transform(y_val_full)
y_test_enc_full  = le.transform(y_test_full)

num_classes = len(le.classes_)

print("Class Labels:", le.classes_)
print("Number of Classes:", num_classes)
# Print the distribution of class labels for each dataset
def print_label_distribution(y, dataset_name):
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"Label distribution for {dataset_name}: {distribution}")

# print_label_distribution(y_train_simple, "Training Set (Simple Features)")
# print_label_distribution(y_val_simple, "Validation Set (Simple Features)")
# print_label_distribution(y_test_simple, "Test Set (Simple Features)")

# print_label_distribution(y_train_full, "Training Set (Full Features)")
# print_label_distribution(y_val_full, "Validation Set (Full Features)")
# print_label_distribution(y_test_full, "Test Set (Full Features)")
# -------------------------------
# Step 4. Create PyTorch Datasets and DataLoaders
# -------------------------------
batch_size = 512

def create_dataloader(X, y, batch_size, shuffle=True):
    # Check for NaNs in the input data
    assert not np.isnan(X).any(), "Input features contain NaNs"
    assert not np.isnan(y).any(), "Labels contain NaNs"

    assert not np.isinf(X).any(), "Input features contain infinite values"
    assert not np.isinf(y).any(), "Labels contain infinite values"

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    print(X_tensor.shape, y_tensor.shape)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# DataLoaders for simple (basic) features
train_loader_simple = create_dataloader(X_train_simple, y_train_enc_simple, batch_size=batch_size)
val_loader_simple = create_dataloader(X_val_simple, y_val_enc_simple, batch_size=batch_size)
test_loader_simple = create_dataloader(X_test_simple, y_test_enc_simple, batch_size=batch_size)

# DataLoaders for full features
train_loader_full = create_dataloader(X_train_full, y_train_enc_full, batch_size=batch_size)
val_loader_full = create_dataloader(X_val_full, y_val_enc_full, batch_size=batch_size)
test_loader_full = create_dataloader(X_test_full, y_test_enc_full, batch_size=batch_size)

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
        nans = False
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Check for NaNs in the outputs
        if torch.isnan(outputs).any():
            print("NaNs detected in model outputs")
            print("Inputs:", inputs)
            print("Outputs:", outputs)
            nans = True
        loss = criterion(outputs, labels)
        
        # Check for NaNs in the loss
        if torch.isnan(loss).any():
            print("NaNs detected in loss")
            print("Outputs:", outputs)
            print("Labels:", labels)
            print("Loss:", loss)
            nans = True
        
        if nans:
            print("Exiting due to NaNs...")
            return np.nan
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, accuracy



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
num_epochs = 100

# Create two models: one for basic features and one for full features.
model_simple = MLP(input_dim=X_train_simple.shape[1], num_classes=num_classes).to(device)
model_full   = MLP(input_dim=X_train_full.shape[1], num_classes=num_classes).to(device)

# Define loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer_simple = optim.Adam(model_simple.parameters(), lr=0.001)
optimizer_full   = optim.Adam(model_full.parameters(), lr=0.001)

# Calculate gamma for exponential decay to 1/100 over the entire training period
gamma = (1/100) ** (1/num_epochs)

# Define learning rate schedulers
scheduler_simple = lr_scheduler.ExponentialLR(optimizer_simple, gamma=gamma)
scheduler_full = lr_scheduler.ExponentialLR(optimizer_full, gamma=gamma)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Train the model with simple features
    train_loss_simple = train_model(model_simple, train_loader_simple, criterion, optimizer_simple, device)
    val_loss_simple, val_acc_simple = evaluate_model(model_simple, val_loader_simple, criterion, device)
    
    # Train the model with full features
    train_loss_full = train_model(model_full, train_loader_full, criterion, optimizer_full, device)
    val_loss_full, val_acc_full = evaluate_model(model_full, val_loader_full, criterion, device)
    
    # Step the learning rate schedulers
    scheduler_simple.step()
    scheduler_full.step()
    
    # Log the performance
    with open(csv_file_simple, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss_simple, val_loss_simple, val_acc_simple])
    
    with open(csv_file_full, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss_full, val_loss_full, val_acc_full])

    print(f"Simple Model - Train Loss: {train_loss_simple:.4f}, Val Loss: {val_loss_simple:.4f}, Val Acc: {val_acc_simple:.4f}")
    print(f"Full Model - Train Loss: {train_loss_full:.4f}, Val Loss: {val_loss_full:.4f}, Val Acc: {val_acc_full:.4f}")

print("Training complete.")


def evaluate_model_test(model, test_files, use_additional_features,outdir):
    results = []
    os.makedirs(outdir, exist_ok = True)
    for file in test_files:
        df = pd.read_csv(file)
        
        # Drop the "index" column if it exists
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        
        # If the dataframe is empty after dropping NaNs, skip this file.
        if df.empty:
            print('  No valid data in the file. Skipping...')
            continue
        
        # Extract the label from the filename.
        base_name = os.path.basename(file)
        label = base_name.split("_")[0]
        
        # Select features based on the flag
        if use_additional_features:
            # Use all features (10 basic features + 3 additional)
            feature_cols = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                            'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4',
                            'delta', 'rel_delta']
        else:
            # Use only the basic 10 features
            feature_cols = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                            'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4']
        
        # Ensure the dataframe has the required columns
        df = df[feature_cols]
        X = df.values.astype(np.float32)  # ensure data is float32
        y = np.full((X.shape[0],), label)
        
        # Remove rows with NaN or infinite values
        nan_mask = np.isnan(X).any(axis=1)
        inf_mask = np.isinf(X).any(axis=1)
        invalid_mask = nan_mask | inf_mask
        if invalid_mask.any():
            X = X[~invalid_mask]
            y = y[~invalid_mask]
            print(f"  Removed {invalid_mask.sum()} rows with NaN or infinite values.")
            print(f"  Processed shape: {X.shape}")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y = le.transform(y)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Make predictions
        batch_size = 512
        predictions = []
        model.eval()
        print('Total number of samples', len(X_tensor))
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size].to(device)
                batch_predictions = model(batch_X)
                
                predictions.append(batch_predictions.cpu())
        predictions = torch.cat(predictions, dim=0)
        print('Predictions shape', predictions.shape)

        predictions = torch.nn.Softmax(dim=1)(predictions)
        print('Predictions shape after softmax', predictions.shape)

        # Save results
        for i in range(len(predictions)):
            result = {'id': i, 'truth': y_tensor[i].item(), 'features': 'full' if use_additional_features else 'limited'}
            for j, class_label in enumerate(le.classes_):
                result[f'pred_{class_label}'] = predictions[i][j].item()
            results.append(result)
    
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(os.path.join(outdir,'model_predictions.csv'), index=False)
    print('Predictions saved to model_predictions.csv')

# -------------------------------
# Step 8. Evaluate on the Test Set
# -------------------------------
test_files = glob.glob(os.path.join(test_dir, "*.csv"))
evaluate_model_test(model_full, test_files, use_additional_features=True, outdir = os.path.join(run_dir,'gromov_features_predictions'))
evaluate_model_test(model_simple, test_files, use_additional_features=False,outdir = os.path.join(run_dir,'base_features_predictions'))
