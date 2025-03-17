import argparse
import h5py
import numpy as np
import torch
import geoopt

def main():
    parser = argparse.ArgumentParser(
        description="Update an HDF5 file by computing the graph distance matrix for each sample."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the HDF5 file to update (e.g., output_data/manifold_data.h5)")
    args = parser.parse_args()

    # Open the file in read/write mode
    with h5py.File(args.input_file, 'a') as f:
        # Check for the 'datasets' dataset
        if 'datasets' not in f:
            print("Error: 'datasets' not found in the file.")
            return
        if 'cur_graph_dist_matrices' in f:
            print("Error: 'cur_graph_dist_matrices' already exists in the file.")
            return
        # Load the datasets array; assume shape is (n_samples, 400, 2)
        datasets = f['datasets'][:]  
        n_samples = datasets.shape[0]
        print(f"Found {n_samples} samples. Each sample has shape {datasets.shape[1:]}.")

        # Prepare a list to hold the computed distance matrices
        graph_dist_matrices = []

        # Create a PoincareBall instance with c=1
        poincare_geom = geoopt.PoincareBall(c=1)

        # Loop through each sample, compute the pairwise distance matrix
        for i in range(n_samples):
            # Convert sample to torch tensor (shape: 400 x 2)
            inputs = torch.tensor(datasets[i], dtype=torch.float32)
            
            # Compute the pairwise distance matrix (shape: 400 x 400)
            dist_matrix = poincare_geom.dist_matrix(inputs, inputs)
            graph_dist_matrices.append(dist_matrix.numpy())
            if (i+1) % 10 == 0 or i == n_samples-1:
                print(f"Processed {i+1}/{n_samples} samples.")

        # Stack all distance matrices into one numpy array (shape: n_samples x 400 x 400)
        graph_dist_matrices = np.stack(graph_dist_matrices)

        # If a dataset with the same name already exists, remove it
        if 'graph_dist_matrices' in f:
            del f['graph_dist_matrices']
        # Create a new dataset in the file with the computed distance matrices
        f.create_dataset('graph_dist_matrices', data=graph_dist_matrices)
        print("The file has been updated with the 'graph_dist_matrices' dataset.")

if __name__ == "__main__":
    main()
