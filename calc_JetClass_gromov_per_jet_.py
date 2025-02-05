import torch
import torch.nn.functional as F
import os
import numpy as np
from dataloader import read_file
import csv

import awkward as ak
import pandas as pd
import sys
import energyflow as ef
import os
from scipy.stats import wasserstein_distance
from dataloader import *


weaver_core_path = os.path.abspath("../weaver-core/weaver")
sys.path.insert(0, weaver_core_path)


def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][None, :]
    col = dismat[:, p][:, None]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
    return (maxmin - XY_p).max()

def compute_rel_delta_mean(emb_c, num_samples=100, sample_size=100):
    result = []
    for _ in range(num_samples):
        idx = torch.randperm(len(emb_c))[:sample_size]
        emb_cur = emb_c[idx]
        dists = torch.cdist(emb_cur, emb_cur)
        delta = delta_hyp(dists)
        diam = dists.max()
        rel_delta = (2 * delta) / diam
        result.append(rel_delta)
    rel_delta_mean = torch.tensor(result).mean().item()
    rel_delta_std = torch.tensor(result).std().item()
    return rel_delta_mean,rel_delta_std

def main():
    JetClass_data_path = '/n/holystore01/LABS/iaifi_lab/Lab/nswood/JetClass/Pythia/test_20M/'
    JetClass_data_files = os.listdir(JetClass_data_path)
    JetClass_data_files = [JetClass_data_path + file for file in JetClass_data_files if '100.root' in file]
    print(JetClass_data_files)
    output_dir = 'JetClass_gromov_delta_results_EMD'
    os.makedirs(output_dir, exist_ok=True)
    for file in JetClass_data_files:
        
        if 'ZToQQ' not in file:
            continue
        print(f"Reading file {file.split('/')[-1]}")
        x_particles, x_jets, y = read_file(file,
                                           particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
                                           jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],)
        print(f"Finished reading file {file.split('/')[-1]}")
        print(f"x_particles shape: {x_particles.shape}")
        print(f"x_jets shape: {x_jets.shape}")
        print(f"y shape: {y.shape}")
        num_jets, _, num_particles = x_particles.shape
        all_part_pt = x_particles[:,0,:]
        all_part_eta = x_particles[:,1,:]
        all_part_phi = x_particles[:,2,:]
        all_part_energy = x_particles[:,3,:]

        csv_name =  file.split('/')[-1].replace('.root', '_gromov_delta_results.csv')
        output_csv = os.path.join(output_dir, csv_name)
        
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(["index", "jet_pt", "jet_eta","jet_nparts","delta", "rel_delta", "c"])

        
        for i in range(num_jets):

            
            jet_pt, jet_eta, jet_phi, jet_energy = x_jets[i]

            # Extract the columns from the DataFrame
            part_energy = all_part_energy[i]
            part_pt = all_part_pt[i]
            part_eta = all_part_eta[i]
            part_phi = all_part_phi[i]
            
            mask = part_energy > 0

            part_energy = part_energy[mask]
            part_pt = part_pt[mask]
            part_eta = part_eta[mask]
            part_phi = part_phi[mask]

            

            n_parts = len(part_energy)


            # Stack the columns to form an nx4 numpy array
            four_momentum_np = np.stack((part_energy, part_eta, part_phi), axis=-1)
            
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
            # print(dists)

            delta = delta_hyp(dists)
            diam = dists.max()
            rel_delta = (2 * delta) / diam

            # Calculate c based on relative delta mean
            c = (0.144 / rel_delta) ** 2
            
            # write i, jet_pt,jet_eta, jet_label, rel_delta, c to a csv output
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i, jet_pt, jet_eta, n_parts,delta.item(), rel_delta.item(), c.item()])
            
            # Print results
            if i % 10 == 0:
                print(f"Jet {i}:δ = {delta:.3f}, relative δ = {rel_delta:.3f}, c = {c:.3f}")

       
        
    print('Finished reading')


if __name__ == "__main__":
    main()


   
   
