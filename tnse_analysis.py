import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_and_concatenate(folder_path, name, num_files):
    """
    Load and concatenate dataset parts from the specified folder.
    
    Args:
        folder_path (str): Path to the dataset folder.
        name (str): Name of the dataset (e.g., "HToWW4Q").
        num_files (int): Number of files to load.

    Returns:
        dict: Concatenated dataset.
    """
    concatenated_data = None
    for i in range(num_files):
        file_path = f"{folder_path}/part_output_{name}_{i}.pt"
        if os.path.exists(file_path):
            data = torch.load(file_path, map_location=torch.device('cpu'))
            if concatenated_data is None:
                concatenated_data = data
            else:
                concatenated_data['tan_space_parts'] = torch.cat(
                    (concatenated_data['tan_space_parts'], data['tan_space_parts']), dim=0
                )
                concatenated_data['man_parts'] = torch.cat(
                    (concatenated_data['man_parts'], data['man_parts']), dim=0
                )
                for key in data['X']:
                    concatenated_data['X'][key] = torch.cat(
                        (concatenated_data['X'][key], data['X'][key]), dim=0
                    )
        else:
            print(f"File not found: {file_path}")
    return concatenated_data

def uniform_sampling(data, pt_values, particle_indices, num_bins=20, total_points=1000):
    """
    Perform uniform sampling across bins of pT values.

    Args:
        data (np.ndarray): Data to sample from.
        pt_values (np.ndarray): pT values for binning.
        particle_indices (np.ndarray): Particle type labels.
        num_bins (int): Number of bins to divide pT values.
        total_points (int): Total number of points to sample.

    Returns:
        tuple: Subset of data, normalized pT values, and particle indices.
    """
    points_per_bin = total_points // num_bins
    bins = np.linspace(pt_values.min(), pt_values.max(), num_bins + 1)
    bin_indices = np.digitize(pt_values, bins)
    selected_indices = []

    for bin_idx in range(1, num_bins + 1):  # np.digitize is 1-indexed
        indices_in_bin = np.where(bin_indices == bin_idx)[0]
        np.random.shuffle(indices_in_bin)
        selected_indices.extend(indices_in_bin[:points_per_bin])

    selected_indices = np.random.choice(selected_indices, size=int(total_points/2), replace=False)
    return data[selected_indices], pt_values[selected_indices], particle_indices[selected_indices]

def tnse_analysis(R4_folder, H4_folder, dataset_name, num_files=1, output_dir='tsne_output'):
    """
    Perform t-SNE analysis for R^4 and H^4 datasets and plot the embeddings.
    
    Args:
        R4_folder (str): Folder containing the R^4 dataset.
        H4_folder (str): Folder containing the H^4 dataset.
        dataset_name (str): Name of the dataset (e.g., "HToWW4Q").
        num_files (int): Number of files to load.
        output_dir (str): Directory to save the output plots.
    """
    # Load datasets
    print("Loading datasets...")
    R4_data = load_and_concatenate(R4_folder, dataset_name, num_files)
    H4_data = load_and_concatenate(H4_folder, dataset_name, num_files)
    print("Datasets loaded successfully!")

    # Prepare R^4 and H^4 data
    R4_embeddings = R4_data["man_parts"].reshape(-1, R4_data["man_parts"].shape[-1]).numpy()
    H4_embeddings = H4_data["man_parts"].reshape(-1, H4_data["man_parts"].shape[-1]).numpy()

    # Normalize pT for coloring
    pt = H4_data['X']['pf_features'].permute(0, 2, 1)[:, :, 0].flatten().numpy()
    H4_normalized_pt = (pt - pt.min()) / (pt.max() - pt.min())
#     print(pt.shape)

    # Extract particle type labels
    hot_encoded = H4_data['X']['pf_features'].permute(0, 2, 1)[:, :, -5:]  # One-hot encoding
    H4_particle_indices = torch.argmax(hot_encoded, dim=-1).numpy().flatten()
#     print(particle_indices.shape)
    
    pt = R4_data['X']['pf_features'].permute(0, 2, 1)[:, :, 0].flatten().numpy()
    R4_normalized_pt = (pt - pt.min()) / (pt.max() - pt.min())
#     print(pt.shape)

    # Extract particle type labels
    hot_encoded = R4_data['X']['pf_features'].permute(0, 2, 1)[:, :, -5:]  # One-hot encoding
    R4_particle_indices = torch.argmax(hot_encoded, dim=-1).numpy().flatten()
#     print(particle_indices.shape)
    
    
    # Uniform sampling over pT bins
    H4_embeddings, H4_normalized_pt, H4_particle_indices = uniform_sampling(
        H4_embeddings, H4_normalized_pt, H4_particle_indices, num_bins=10, total_points=50000
    )
    
    R4_embeddings, R4_normalized_pt, R4_particle_indices = uniform_sampling(
        R4_embeddings, R4_normalized_pt, R4_particle_indices, num_bins=10, total_points=50000
    )

    # Perform t-SNE
    print("Calculating t-SNE for R^4...")
    tsne_r4 = TSNE(n_components=2, perplexity=20, n_iter=1000, random_state=42)
    embeddings_r4 = tsne_r4.fit_transform(R4_embeddings)
    
    print("Calculating t-SNE for H^4...")
    tsne_h4 = TSNE(n_components=2, perplexity=20, n_iter=1000, random_state=42)
    embeddings_h4 = tsne_h4.fit_transform(H4_embeddings)

    # Plotting helper functions
    def plot_embeddings(embeddings, normalized_pt, title, filename):
        plt.figure(figsize=(8, 8))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=plt.cm.viridis(normalized_pt), s=20, alpha=0.7)
        plt.title(title)
        plt.axis('off')
        plt.colorbar(label='Normalized pT')
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def plot_particle_type_embeddings(embeddings, particle_indices, title, filename):
        particle_types = ['Charged Hadron', 'Neutral Hadron', 'Photon', 'Electron', 'Muon']
        plt.figure(figsize=(8, 8))
        for i in np.unique(particle_indices):
            mask = particle_indices == i
            plt.scatter(embeddings[mask][:, 0], embeddings[mask][:, 1], label=particle_types[int(i)], s=20, alpha=0.7)
        plt.title(title)
        plt.axis('off')
        plt.legend()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save embeddings
    print("Plotting results...")
    plot_embeddings(embeddings_r4, R4_normalized_pt, 't-SNE Embedding (R^4)', 'tsne_r4_pt.png')
    plot_embeddings(embeddings_h4, H4_normalized_pt, 't-SNE Embedding (H^4)', 'tsne_h4_pt.png')

    plot_particle_type_embeddings(embeddings_r4, R4_particle_indices, 't-SNE Embedding (R^4) Particle Types', 'tsne_r4_particle_types.png')
    plot_particle_type_embeddings(embeddings_h4, H4_particle_indices, 't-SNE Embedding (H^4) Particle Types', 'tsne_h4_particle_types.png')

    print(f"Plots saved in the '{output_dir}' directory.")

# Example usage
if __name__ == "__main__":
    R4_folder = "/n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241122-231227_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_4_h4q_R_4_3_new_data/predict_output"
    H4_folder = "/n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241122-225952_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_4_h4q_H_4_2_new_data/predict_output"
    dataset_name = "HToWW4Q"
    tnse_analysis(R4_folder, H4_folder, dataset_name, num_files=10)
