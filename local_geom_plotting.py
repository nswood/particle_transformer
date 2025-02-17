# %%
import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][None, :]
    col = dismat[:, p][:, None]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
    return (maxmin - XY_p).max()



def find_neighbors(four_momentum_tensor, dists, index=None, r_percent = 0.05):
    # Randomly select a point
    num_points = four_momentum_tensor.shape[0]
    if index is None:
        index = random.randint(0, num_points - 1)
    selected_point = four_momentum_tensor[index]
    r = r_percent *selected_point[0]
    # Find neighbors within distance r
    neighbors_indices = torch.where(dists[index] <= r)[0]

    neighbors = four_momentum_tensor[neighbors_indices]

    return selected_point, neighbors, neighbors_indices





def find_neighbors_knn(four_momentum_tensor, dists, index=None, k=5):
    # Randomly select a point
    num_points = four_momentum_tensor.shape[0]
    if index is None:
        index = random.randint(0, num_points - 1)
    selected_point = four_momentum_tensor[index]
    
    # Find the indices of the k-nearest neighbors
    neighbors_indices = torch.topk(dists[index], k, largest=False).indices

    neighbors = four_momentum_tensor[neighbors_indices]

    return selected_point, neighbors, neighbors_indices



parquet_file = '/n/holystore01/LABS/iaifi_lab/Lab/nswood/QuarkGluon/train_file_0.parquet'
# config_file = 'data/QuarkGluon/qg_kinpid.yaml'
print('Starting reading QG')

# Load data from Parquet file
df = pd.read_parquet(parquet_file)
# print(df.columns)
print('Finished reading QG')
jet_pt, jet_eta,jet_phi, jet_label = df['jet_pt'], df['jet_eta'],df['jet_phi'],df['label']
four_momentum = df[['part_energy', 'part_px', 'part_py', 'part_pz']]
part_eta = df['part_deta']
part_phi = df['part_dphi']


all_k = [3, 5, 8, 12]

results = []
for i in range(200):
    cur_data = four_momentum.iloc[i]
    cur_jet_data = [jet_pt[i], jet_eta[i], jet_label[i]]

    # Extract the columns from the DataFrame
    part_energy = cur_data['part_energy']
    part_px = cur_data['part_px']
    part_py = cur_data['part_py']
    part_pt = np.sqrt(part_px**2 + part_py**2)
    n_parts = len(part_pt)


    # Stack the columns to form an nx4 numpy array
    four_momentum_np = np.stack((part_energy, part_eta.iloc[i] , part_phi.iloc[i]), axis=-1)

    # Convert the numpy array to a torch tensor
    four_momentum_tensor = torch.tensor(four_momentum_np)

    # Extract energy, eta, phi
    energies = four_momentum_tensor[:, 0]
    # normalized_energies = energies / energies.sum()
    # energies = normalized_energies

    # print(normalized_energies)
    etas = four_momentum_tensor[:, 1]
    phis = four_momentum_tensor[:, 2]

    # Compute pairwise energy differences |E_i - E_j|
    energy_diffs = torch.abs(energies[:, None] - energies[None, :])

    # Compute pairwise angular distances ΔR_ij = sqrt((η_i - η_j)^2 + (φ_i - φ_j)^2)
    delta_eta = etas[:, None] - etas[None, :]
    delta_phi = phis[:, None] - phis[None, :]

    # Ensure phi differences wrap around correctly (handle 2pi periodicity)
    delta_phi = torch.remainder(delta_phi + np.pi, 2 * np.pi) - np.pi

    # Calculate ΔR
    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

    # Calculate EMD-inspired distance: E_diff * ΔR / R (where R is a scale factor, e.g., jet radius)
    R = 1  # Example jet radius
    dists = (energy_diffs * delta_R) / R
    

    for j in range(len(four_momentum_tensor)):
        n_parts = len(four_momentum_tensor)
        if n_parts < 32:
            continue
        for k in all_k:
            # print(k)
            selected_point, neighbors, neighbors_indices = find_neighbors_knn(four_momentum_tensor, dists, index=j, k=k)

            # Extract the submatrix for the neighbors
            neighbors_dists = dists[neighbors_indices][:, neighbors_indices]

            delta = delta_hyp(neighbors_dists)
            diam = neighbors_dists.max()
            rel_delta = (2 * delta) / diam

            # Calculate c based on relative delta mean
            c = (0.144 / rel_delta) ** 2

            # Save the results
            results.append({
                'selected_point_energy': selected_point[0].item(),
                'selected_point_eta': selected_point[1].item(),
                'selected_point_phi': selected_point[2].item(),
                'num_parts': n_parts,
                'k': k,
                'delta': delta.item(),
                'rel_delta': rel_delta.item(),
                'c': c.item(),
                'jet_id':i
            })

    # Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Create directory for plots if it doesn't exist
plot_dir = 'local_geom_plots'
os.makedirs(plot_dir, exist_ok=True)

# Plot comparison of k, delta, and selected_point_energy
for k in all_k:
    plt.figure(figsize=(10, 8))
    subset = results_df[results_df['k'] == k]
    plt.scatter(subset['selected_point_energy'], subset['delta'], c=subset['num_parts'], cmap='viridis')
    plt.xlabel('Particle Energy',fontsize = 18)
    plt.ylabel(f'EMD Gromov-$\delta$ (MeV)',fontsize = 18)
    plt.title(f'QvG Local Jet Topology Estimate from EMD Gromov-$\delta$ (MeV) with k={k} nearest neighbors',fontsize = 14)
    plt.colorbar(label='Jet Number of Particles')
    plt.savefig(os.path.join(plot_dir, f'GQ_k_{k}_delta_vs_energy.png'))
    plt.close()

bins = np.linspace(results_df['delta'].min(), results_df['delta'].max(), 50)

# We'll accumulate normalized histograms per jet
sum_of_histograms = np.zeros(len(bins) - 1)
n_jets = results_df['jet_id'].nunique()

# Loop over jets
for jet_id, group in results_df.groupby('jet_id'):
    # Compute histogram counts for this jet
    hist_counts, _ = np.histogram(group['delta'], bins=bins)
    # Normalize so the area (sum of counts) is 1 for each jet
    hist_counts = hist_counts / hist_counts.sum()
    # Accumulate the normalized histograms
    sum_of_histograms += hist_counts

# Now average them
avg_hist = sum_of_histograms / n_jets

# Bin centers for plotting
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Plot the average histogram
plt.figure()
plt.bar(bin_centers, avg_hist, width=(bins[1]-bins[0]), alpha=0.7, align='center')
plt.xlabel(r"$\delta$")
plt.ylabel("Average normalized density")
plt.xscale('log')
plt.title("Average distribution of $\delta$ across all jets")
plt.savefig(os.path.join(plot_dir, f'GQ_delta_dist.png'))
plt.close()

from scipy.optimize import curve_fit

# ---------------------------------------------------------------------
# 3. Plot each jet's histogram on the same figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Choose some bin edges for the histogram
# You can pick based on your data range or auto (e.g. np.linspace).
all_delta = results_df['delta'].values
num_bins = 50
bin_edges = np.linspace(all_delta.min(), all_delta.max(), num_bins)

# Plot each jet's delta distribution
for jet_id, group in results_df.groupby('jet_id'):
    ax.hist(
        group['delta'], 
        bins=bin_edges,
        histtype='step',
        alpha=0.3, 
        density=True,  # or True if you prefer normalized hist
        edgecolor='blue',

    )

# ---------------------------------------------------------------------
# 4. Fit a single "best fit" line (Gaussian) over ALL data
# ---------------------------------------------------------------------
# 4a. Compute the total histogram over all jets
counts, edges = np.histogram(all_delta, bins=bin_edges, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])

# 4b. Use curve_fit to find the best-fit Gaussian parameters
#     We'll use a simple initial guess: 
#     A ~ max(counts), mu ~ mean of all data, sigma ~ std of all data



# We'll try curve_fit with our two-sided Gaussian function:
# import numpy as np
from scipy.optimize import curve_fit

# Define the exponential decay function
def exp_decay(x, A, b, c):
    return A * np.exp(-b * x) + c

# Set ranges for random initial guesses
param_bounds = {
    'A': (0.1, 10),   # Amplitude range
    'b': (0.01, 2),   # Decay rate range
    'c': (-1, 1),     # Offset range
}


best_popt = None
lowest_residual = np.inf

num_trials = 1000
x_fit = np.linspace(edges[0], edges[-1], 500)

for _ in range(num_trials):
    try:
        # Generate random initial guesses
        p0 = [
            np.random.uniform(*param_bounds['A']),
            np.random.uniform(*param_bounds['b']),
            np.random.uniform(*param_bounds['c']),
        ]
        
        # Perform the fit
        popt, pcov = curve_fit(exp_decay, bin_centers, counts, p0=p0)
        
        # Calculate residuals (sum of squared differences)
        residuals = np.sum((counts - exp_decay(bin_centers, *popt))**2)
        
        # Update best fit if current fit is better
        if residuals < lowest_residual:
            lowest_residual = residuals
            best_popt = popt
            
    except RuntimeError:
        # Skip fits that fail
        continue

# Plot the best fit if one was found
if best_popt is not None:
    A_fit, b_fit, c_fit = best_popt
    y_fit = exp_decay(x_fit, A_fit, b_fit, c_fit)
    ax.plot(x_fit, y_fit, 'r-', label='Exponential Decay Fit', linewidth=2)
else:
    print("Failed to find a suitable exponential decay fit after 1000 trials.")


# ---------------------------------------------------------------------
# 6. Figure decorations
# ---------------------------------------------------------------------
ax.set_xlabel(r'$\delta$')
ax.set_ylabel('Counts')
ax.set_title('All Jet Histograms + Best-Fit Gaussian Over ALL Data')
# ax.set_yscale('log')
# If you have many jets, the legend can get long. 
# You might want to show just the best-fit line in the legend,
# or limit the number of jets. For demonstration, let's show all:
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'GQ_all_delta_dist.png'))
plt.close()


parquet_file = '/n/holystore01/LABS/iaifi_lab/Lab/nswood/TopLandscape/val_file.parquet'
# config_file = 'data/QuarkGluon/qg_kinpid.yaml'
print('Starting reading Top')

# Load data from Parquet file
df = pd.read_parquet(parquet_file)
# print(df.columns)
print('Finished reading Top')
jet_pt, jet_eta,jet_phi, jet_label = df['jet_pt'], df['jet_eta'],df['jet_phi'],df['label']
four_momentum = df[['part_energy', 'part_px', 'part_py', 'part_pz']]
part_eta = df['part_deta']
part_phi = df['part_dphi']

# %%
jet_pt, jet_eta,jet_phi, jet_label = df['jet_pt'], df['jet_eta'],df['jet_phi'],df['label']
four_momentum = df[['part_energy', 'part_px', 'part_py', 'part_pz']]
part_eta = df['part_deta']
part_phi = df['part_dphi']

all_k = [3, 5, 8, 12]

results = []
for i in range(5000):
    cur_data = four_momentum.iloc[i]
    cur_jet_data = [jet_pt[i], jet_eta[i], jet_label[i]]

    # Extract the columns from the DataFrame
    part_energy = cur_data['part_energy']
    part_px = cur_data['part_px']
    part_py = cur_data['part_py']
    part_pt = np.sqrt(part_px**2 + part_py**2)
    n_parts = len(part_pt)


    # Stack the columns to form an nx4 numpy array
    four_momentum_np = np.stack((part_energy, part_eta.iloc[i] , part_phi.iloc[i]), axis=-1)

    # Convert the numpy array to a torch tensor
    four_momentum_tensor = torch.tensor(four_momentum_np)

    # Extract energy, eta, phi
    energies = four_momentum_tensor[:, 0]
    # normalized_energies = energies / energies.sum()
    # energies = normalized_energies

    # print(normalized_energies)
    etas = four_momentum_tensor[:, 1]
    phis = four_momentum_tensor[:, 2]

    # Compute pairwise energy differences |E_i - E_j|
    energy_diffs = torch.abs(energies[:, None] - energies[None, :])

    # Compute pairwise angular distances ΔR_ij = sqrt((η_i - η_j)^2 + (φ_i - φ_j)^2)
    delta_eta = etas[:, None] - etas[None, :]
    delta_phi = phis[:, None] - phis[None, :]

    # Ensure phi differences wrap around correctly (handle 2pi periodicity)
    delta_phi = torch.remainder(delta_phi + np.pi, 2 * np.pi) - np.pi

    # Calculate ΔR
    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

    # Calculate EMD-inspired distance: E_diff * ΔR / R (where R is a scale factor, e.g., jet radius)
    R = 1  # Example jet radius
    dists = (energy_diffs * delta_R) / R
    

    for j in range(len(four_momentum_tensor)):
        n_parts = len(four_momentum_tensor)
        if n_parts < 32:
            continue
        for k in all_k:
            # print(k)
            selected_point, neighbors, neighbors_indices = find_neighbors_knn(four_momentum_tensor, dists, index=j, k=k)

            # Extract the submatrix for the neighbors
            neighbors_dists = dists[neighbors_indices][:, neighbors_indices]

            delta = delta_hyp(neighbors_dists)
            diam = neighbors_dists.max()
            rel_delta = (2 * delta) / diam

            # Calculate c based on relative delta mean
            c = (0.144 / rel_delta) ** 2

            # Save the results
            results.append({
                'selected_point_energy': selected_point[0].item(),
                'selected_point_eta': selected_point[1].item(),
                'selected_point_phi': selected_point[2].item(),
                'num_parts': n_parts,
                'k': k,
                'delta': delta.item(),
                'rel_delta': rel_delta.item(),
                'c': c.item(),
                'jet_id':i
            })

    # Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Create directory for plots if it doesn't exist
plot_dir = 'local_geom_plots'
os.makedirs(plot_dir, exist_ok=True)

# Plot comparison of k, delta, and selected_point_energy
for k in all_k:
    plt.figure(figsize=(10, 8))
    subset = results_df[results_df['k'] == k]
    plt.scatter(subset['selected_point_energy'], subset['delta'], c=subset['num_parts'], cmap='viridis')
    plt.xlabel('Particle Energy',fontsize = 18)
    plt.ylabel(f'EMD Gromov-$\delta$ (MeV)',fontsize = 18)
    plt.title(f'Top Local Jet Topology Estimate from EMD Gromov-$\delta$ (MeV) with k={k} nearest neighbors',fontsize = 14)
    plt.colorbar(label='Jet Number of Particles')
    plt.savefig(os.path.join(plot_dir, f'Top_k_{k}_delta_vs_energy.png'))
    plt.close()

