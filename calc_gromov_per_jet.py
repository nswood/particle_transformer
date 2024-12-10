import torch
import torch.nn.functional as F
import os
import numpy as np
from dataloader import read_file
import csv

import awkward as ak
import pandas as pd
import sys
import os

weaver_core_path = os.path.abspath("../weaver-core/weaver")
sys.path.insert(0, weaver_core_path)
from utils.dataset import _preprocess, DataConfig

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

    # File paths
    parquet_file = '/n/holystore01/LABS/iaifi_lab/Lab/nswood/TopLandscape/test_file.parquet'
    config_file = 'data/TopLandscape/top_kin.yaml'

    # Load data from Parquet file
    df = pd.read_parquet(parquet_file)
    
    # Convert DataFrame to a list of dictionaries
    records = df.to_dict('records')

    # Create the Awkward Array
    ak_array = ak.from_iter(records)


    # Load data configuration
    data_config = DataConfig.load(config_file)
    

    # Define options for preprocessing
    options = {
        'training': False,        # True for training, False for testing
        'shuffle': False,         # Shuffle the data
        'reweight': False,       # Reweighting is optional
        'up_sample': True,       # Upsample if necessary
        'weight_scale': 1,       # Weight scaling factor
        'max_resample': 10,      # Maximum resampling iterations
    }

    processed_data, indices = _preprocess(ak_array, data_config, options)
    pf_data = processed_data['_pf_features'].transpose(0,2,1)
    pf_mask = processed_data['_pf_mask'].transpose(0,2,1)
    jet_pt, jet_eta, jet_label = processed_data['jet_pt'], processed_data['jet_eta'],processed_data['_label_']
    
    # make csv to store gromov delta calculations for all
    output_csv = "top_gromov_delta_results.csv"
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["index", "jet_pt", "jet_eta", "jet_label","jet_nparts", "rel_delta", "c"])

    
    for i in range(len(pf_data)):
        cur_data = pf_data[i]
        cur_mask = pf_mask[i,:,0].astype(bool)
        cur_data= cur_data[cur_mask]
        cur_jet_data = [jet_pt[i], jet_eta[i], jet_label[i]]
        n_parts = len(cur_data)
        # Compute relative delta mean
        dists = torch.cdist(torch.tensor(cur_data), torch.tensor(cur_data))
        delta = delta_hyp(dists)
        diam = dists.max()
        rel_delta = (2 * delta) / diam

        # Calculate c based on relative delta mean
        c = (0.144 / rel_delta) ** 2
        
        # write i, jet_pt,jet_eta, jet_label, rel_delta, c to a csv output
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, jet_pt[i], jet_eta[i], jet_label[i],n_parts, rel_delta.item(), c.item()])
        
        # Print results
        print(f"Jet {i}: δ = {rel_delta:.3f}, c = {c:.3f}")

if __name__ == "__main__":
    main()


   
   
