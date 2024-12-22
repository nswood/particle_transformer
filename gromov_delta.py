import torch
import torch.nn.functional as F
import os
import numpy as np
from dataloader import read_file

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

def compute_rel_delta_mean_batched(emb, num_samples=250, sample_size=10):
    result = []
    
    batch_size = emb.shape[0]  # The first dimension represents the batch
    for _ in range(num_samples):
        # Randomly select an element from the batch
        batch_idx = torch.randint(0, batch_size, (1,)).item()
        emb_sample = emb[batch_idx]  # Shape: (particles, features)
        
        # Remove zero-padded particles (i.e., those with all zero features)
        non_zero_mask = (emb_sample != 0).any(dim=1)  # Boolean mask to select non-zero rows
        emb_trimmed = emb_sample[non_zero_mask]
        emb_trimmed[:,0:2] = np.log(np.abs(emb_trimmed[:,0:2]))
        emb_trimmed[:,0] = (emb_trimmed[:,0]-1.7)/5
        emb_trimmed[:,1] = (emb_trimmed[:,0]-2)/5
        

#         print(emb_sample)
        
        # If the sample is too small after trimming, skip the iteration
        if len(emb_trimmed) < sample_size:
            continue
        
        # Randomly sample a subset of particles (based on sample_size)
        idx = torch.randperm(len(emb_trimmed))[:sample_size]
        emb_cur = emb_trimmed[idx]
        
        # Compute the pairwise distances and rel_delta
        dists = torch.cdist(emb_cur, emb_cur)
        delta = delta_hyp(dists)
        diam = dists.max()
        rel_delta = (2 * delta) / diam
        result.append(rel_delta)
    
    # Compute the mean of relative delta values
    
    rel_delta_mean = torch.tensor(result).mean().item()
    rel_delta_std = torch.tensor(result).std().item()
    
    return rel_delta_mean,rel_delta_std

def main():
    DATADIR = '/n/holystore01/LABS/iaifi_lab/Lab/nswood/JetClass'
    SAMPLE_TYPE = 'Pythia'
    
    paths = [
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/HToBB_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/HToCC_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/HToGG_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/HToWW2Q1L_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/HToWW4Q_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/TTBar_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/TTBarLep_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/WToQQ_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/ZToQQ_000.root",
        f"{DATADIR}/{SAMPLE_TYPE}/train_100M/ZJetsToNuNu_000.root",
    ]
    
    processes = ['Hbb', 'Hcc', 'Hgg', 'Hqql', 'H4q','Tbqq', 'Tbl','Wqq', 'Zqq','QCD']

    for i in range(10):
        

        features, x_jet, y = read_file(paths[i], particle_features = ['part_pt', 'part_energy','part_deta','part_dphi','part_charge','part_isChargedHadron','part_isNeutralHadron','part_isPhoton','part_isElectron','part_isMuon'])
#         features, x_jet, y = read_file(paths[i], particle_features = ['part_pt', 'part_energy','part_deta','part_dphi'])
        
        features = torch.tensor(features).permute(0,2,1)

        print(f"Processing {processes[i]}")

        # Get the embeddings for the current process
        features_c = features[0:50000]
#         features_c = features_c.reshape(features_c.shape[0] * features_c.shape[1], features_c.shape[2])

        # Compute relative delta mean
        rel_delta_mean,rel_delta_std = compute_rel_delta_mean_batched(features_c)

        # Calculate c based on relative delta mean
        c = (0.144 / rel_delta_mean) ** 2
        c_min =  (0.144 / (rel_delta_mean-rel_delta_std)) ** 2
        c_max =  (0.144 / (rel_delta_mean+rel_delta_std)) ** 2
        # Print results
        print(f"{processes[i]} δ = {rel_delta_mean:.3f}, c = {c:.3f}, c_range = [{c_min},{c_max}]")

if __name__ == "__main__":
    main()


# print(f'all after embeddingclasses')
# emb = features[0:2000]
# emb_c = emb.reshape(emb.shape[0]*emb.shape[1],emb.shape[2])[0:2000]

# result = []
# for i in range(100):
#     idx = torch.randperm(len(emb_c))[:2000]
#     emb_cur = emb_c[idx]
#     dists = torch.cdist(emb_cur, emb_cur)
#     delta = delta_hyp(dists)
#     diam = dists.max()
#     rel_delta = (2 * delta) / diam
#     result.append(rel_delta)
# rel_delta_mean = torch.tensor(result).mean().item()
# c = (0.144 / rel_delta_mean) ** 2

# print(f"δ = {rel_delta_mean:.3f}, c = {c:.3f}")
    
   
   
