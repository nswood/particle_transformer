import os
import numpy as np
import torch
import geoopt
import argparse
import h5py

def manifold_gaussian_pdf(x, mean, std, manifold):
    """
    Compute the probability density of a point x under a manifold Gaussian.
    """
    # Compute the manifold distance between x and the mean
    d = manifold.dist(torch.tensor(x, dtype=torch.float32), mean)
    # Gaussian decay in manifold space
    return np.exp(-d**2 / (2 * std**2))

def sample_manifold_gaussian(n_points, mean, std, manifold, max_attempts=100000):
    """
    Sample n_points from a manifold Gaussian using rejection sampling.
    
    Parameters:
      n_points (int): Number of points to sample.
      mean (torch.Tensor): Mean point in the manifold (shape: (2,)).
      std (float): Standard deviation for the Gaussian decay.
      manifold: A geoopt manifold instance.
      max_attempts (int): Maximum number of sampling attempts.
    
    Returns:
      torch.Tensor: Sampled points of shape (n_points, 2).
    """
    samples = []
    attempts = 0
    while len(samples) < n_points and attempts < max_attempts:
        # Propose a random point in the ambient 2D space.
        
        radius = np.random.uniform(0,1/np.sqrt(np.abs(manifold.k)))
        theta = np.random.uniform(0, 2*np.pi)
        x = radius*np.array([np.cos(theta), np.sin(theta)])
        # Ensure the point lies inside the manifold (ball of radius 1/sqrt(|c|)).
        
        # Compute the acceptance probability
        prob = manifold_gaussian_pdf(x, mean, std, manifold)
        threshold = np.random.uniform(0, 1)
        if prob > threshold:
            samples.append(x)
        attempts += 1

    if len(samples) < n_points:
        print("Warning: Could not generate enough points within max_attempts.")
    samples = np.array(samples)
    return torch.tensor(samples, dtype=torch.float32)


def main():
    print('Generating data...')
    parser = argparse.ArgumentParser(
        description="Generate manifold Gaussian clusters and save the dataset."
    )
    # n_samples here represents the number of points per cluster.
    parser.add_argument("--n_datapoints", type=int, default=25000,
                        help="Number of distinct distributions to generate (default: 25000)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of points per cluster (default: 100)")
    parser.add_argument("--n_centroids", type=int, default=4,
                        help="Number of clusters/centroids per distribution (default: 4)")
    parser.add_argument("--curvature", type=float, default=-1.0,
                        help="Curvature parameter for the manifold (default: -1.0)")
    parser.add_argument("--outdir", type=str, default="output_data",
                        help="Output directory to save the data (default: output_data)")
    parser.add_argument("--file_name", type=str, default="manifold_data",
                        help="Output file name (default: manifold_data)")
    
    args = parser.parse_args()

    # Ensure the output directory exists.
    os.makedirs(args.outdir, exist_ok=True)

    # Setup the manifold with the specified curvature.
    manifold = geoopt.StereographicExact(k=args.curvature)
    radius = manifold.radius
    all_datasets = []
    all_labels = []
    all_tan_datasets = []
    all_centroids = []
    all_tan_centroids = []
    for d in range(args.n_datapoints):
        print(f"Generating distribution {d+1}/{args.n_datapoints}...")
        while True:
            # Step 1: Sample centroids from a zero-mean manifold Gaussian (variance 1.5)
            mean_zero = torch.zeros(2)  # Center at origin
            centroid_var = 1 * radius.numpy()
            centroids = sample_manifold_gaussian(n_points=args.n_centroids, mean=mean_zero, std=centroid_var, manifold=manifold)
            all_centroids.append(centroids)
            tan_centroids = manifold.logmap0(centroids)
            all_tan_centroids.append(tan_centroids)
            # Generate data for each cluster.
            dataset = []
            labels = []
            tan_dataset = []

            # Variable variance for each cluster 
            var_percents = np.random.uniform(0.15, 0.5, args.n_centroids)
            vars = var_percents * radius.numpy()

            success = True
            for i, centroid in enumerate(centroids):
                print(f"  Cluster {i+1}/{args.n_centroids}...")
                points = sample_manifold_gaussian(n_points=args.n_samples, mean=centroid, std=vars[i], manifold=manifold)
                if len(points) < args.n_samples:
                    print(f"  Not enough points generated for cluster {i+1}. Retrying distribution {d+1}...")
                    success = False
                    break
                tan_points = manifold.logmap(points, centroid) + manifold.logmap0(centroid)
                labels.extend([i] * len(points))  # Assign class labels
                tan_dataset.append(tan_points)
                dataset.append(points)

            if success:
                # Convert to torch tensor and append to the list
                dataset = torch.concatenate(dataset)
                tan_dataset = torch.concatenate(tan_dataset)
                labels = torch.tensor(labels)

                # Permute the data
                perm = torch.randperm(len(labels))
                dataset = dataset[perm]
                tan_dataset = tan_dataset[perm]
                labels = labels[perm]

                all_datasets.append(dataset)
                all_labels.append(labels)
                all_tan_datasets.append(tan_dataset)
                break

    # Convert to single array
    all_datasets = np.stack(all_datasets)
    all_labels = torch.stack(all_labels).numpy()
    all_tan_datasets = np.stack(all_tan_datasets)
    all_centroids = torch.stack(all_centroids).numpy()
    all_tan_centroids = torch.stack(all_tan_centroids).numpy()
    print('Data generated successfully!')

    # Save the data.
    output_path = os.path.join(args.outdir, args.file_name+'.h5')
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('datasets', data=all_datasets)
        f.create_dataset('labels', data=all_labels)
        f.create_dataset('tan_datasets', data=all_tan_datasets)
        f.create_dataset('centroids', data=all_centroids)
        f.create_dataset('tan_centroids', data=all_tan_centroids)

if __name__ == "__main__":
    main()
