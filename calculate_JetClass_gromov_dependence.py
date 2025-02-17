import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
from datetime import datetime
import uproot

import torch

path = '/n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/kinpid/PMTrans/'

gromov_paths = '/n/home11/nswood/particle_transformer/JetClass_gromov_delta_results_EMD/test/'

gromov_loaded_data = {}
combined_data = []

for gromov_path in os.listdir(gromov_paths):
    class_name = gromov_path.split('_100')[0]
    data = pd.read_csv(os.path.join(gromov_paths, gromov_path))['delta'].values
    mask = ~np.isnan(data) & ~np.isinf(data)
    data = data[mask]
    gromov_loaded_data[class_name + '_mask'] = mask
    gromov_loaded_data[class_name] = data

    combined_data.extend(data)

combined_data = np.array(combined_data)
deciles = np.percentile(combined_data, np.arange(0, 101, 20))



all_output_folders = glob.glob(path + "*/predict_output")

print(f'Determining Gromov Dependence for {len(all_output_folders)} models')
all_classes = [a.split('_100')[0] for a in os.listdir(gromov_paths)]

# ['HToWW2Q1L', 'HToBB', 'HToWW4Q', 'TTBar', 'ZJetsToNuNu', 'HToGG', 'WToQQ', 'ZToQQ', 'TTBarLep', 'HToCC']
name_reapping ={'HToWW2Q1L':'Hbb', 'HToBB':'Hbb', 'HToWW4Q':'H4q', 'TTBar':'Tbqq', 'ZJetsToNuNu':'QCD', 'HToGG':'Hgg', 'WToQQ':'Wqq', 'ZToQQ':'Zqq', 'TTBarLep':'Tbl', 'HToCC':'Hcc'}

# Looping over each model's prediction[s
for output_folder in all_output_folders:
    output_log_dir = output_folder.split('predict_output')[0]
    # print('Output log dir: ', output_log_dir)
    model_name = output_folder.split('PMTrans_')[1].split('/')[0]
    print(model_name)

    model_results = []

    for class_name in all_classes:
        print('Loading class: ', class_name)
        # Load the model's predictions
        root_file_path = os.path.join(output_folder, f'pred_{class_name}.root')
        if not os.path.exists(root_file_path):
            print(f"File {root_file_path} does not exist.")
            continue

        with uproot.open(root_file_path)['Events;1'] as file:
            
            model_scores = [key for key in file.keys() if 'score_label_' in key]

            combined_model_score = np.array([file[key].array() for key in model_scores])
            prediciton = np.argmax(combined_model_score, axis=0)
            truths = file['_label_'].array()

            prediciton = prediciton[gromov_loaded_data[class_name + '_mask']]
            truths = truths[gromov_loaded_data[class_name + '_mask']]
            gromov_values = gromov_loaded_data[class_name]
            
            correct = np.sum(prediciton == truths)
            incorrect = np.sum(prediciton != truths)

            

         
            for i in range(len(deciles) - 1):
                decile_min = deciles[i]
                decile_max = deciles[i + 1]

                # Filter the data for the current decile
                decile_mask = (gromov_loaded_data[class_name] >= decile_min) & (gromov_loaded_data[class_name] < decile_max)
                decile_indices = np.where(decile_mask)[0]

                decile_correct = np.sum(prediciton[decile_indices] == truths[decile_indices])
                decile_incorrect = np.sum(prediciton[decile_indices] != truths[decile_indices])

                # Append the results to the model_results list
                model_results.append({
                    'Model': model_name,
                    'Class': class_name,
                    'Decile': f'{decile_min}-{decile_max}',
                    'Decile_idx': i,
                    'Correct': decile_correct,
                    'Incorrect': decile_incorrect
                })
                # print(f"Decile {decile_min}-{decile_max}: {decile_correct} correct, {decile_incorrect} incorrect")

    # # Convert the results to a DataFrame and save it to a CSV file
    results_df = pd.DataFrame(model_results)
    results_df.to_csv(os.path.join(output_log_dir, f'{model_name}_gromov_analysis.csv'), index=False)
    print(f"Results saved to {os.path.join(output_log_dir, f'{model_name}_results.csv')}")
          



# print(all_output_folders)