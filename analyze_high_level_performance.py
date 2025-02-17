#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob
from sklearn.preprocessing import LabelEncoder

def load_physics_features(test_files,label_map):
    all_features = []
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
        
        # Use all features (10 basic features + 3 additional)
        feature_cols = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                        'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4',
                        'delta', 'rel_delta']
        
        # Ensure the dataframe has the required columns
        df = df[feature_cols]
        label_enc = label_map[label]
        fit_labels = (label_enc*np.ones(len(df))).astype(int)
        df['label'] = fit_labels
        df['index'] = df.index
        
        all_features.append(df)
    
    return pd.concat(all_features, ignore_index=True)
        

def analyze_model_performance(log_dir, 
                              out_dir="analysis_plots",
                              sample_for_pairplot=1000,
                              label_encoder=None,
                              physical_features=None):
    """
    Loads predictions from two models (simple and full) from CSV files and uses the embedded
    physical feature columns to analyze performance. It assumes that each predictions CSV includes:
      - An "id" column,
      - A "truth" column (with the true label as a numeric value),
      - One or more physical feature columns (e.g., jet_pt, jet_eta, etc.),
      - Prediction probability columns whose names start with "pred_".
    
    The function computes the predicted label (using argmax over the prediction probability columns)
    for each model, marks whether each prediction is correct, and then groups the samples into:
      - "both_correct": both models predicted correctly,
      - "both_incorrect": both models predicted incorrectly,
      - "disagreement": one model was correct while the other was not.
    
    It then creates several plots:
      - A count plot of the three groups,
      - A scatter plot (jet_pt vs. jet_eta, if available),
      - Overlaid KDE histograms for each physical feature,
      - A pairplot of the physical features (sampled if needed),
      - Correlation heatmaps of the physical features for each group.
    
    All generated plots and a combined CSV file are saved to the specified output directory.
    
    Parameters:
      pred_file_simple (str): Path to the CSV file with predictions from the simple model.
      pred_file_full (str): Path to the CSV file with predictions from the full model.
      out_dir (str): Directory where analysis plots and CSV will be saved.
      sample_for_pairplot (int): Maximum number of samples to include in the pairplot.
      label_encoder (LabelEncoder, optional): A pre-fitted LabelEncoder. If None, one is built
                                              from available string labels (if provided in a "true_str" column).
      physical_features (list of str, optional): List of column names for the physical features.
                                                 If not provided, the function will attempt to infer them.
    """
    os.makedirs(out_dir, exist_ok=True)

    label_map = pd.read_csv(os.path.join(log_dir,'label_mapping.csv'))
    label_map = pd.read_csv(os.path.join(log_dir, 'label_mapping.csv'))
    label_map = dict(zip(label_map['class'], label_map['encoded_label']))
        
    pred_file_simple = os.path.join(log_dir, 'base_features_predictions/model_predictions.csv')
    pred_file_full = os.path.join(log_dir, 'gromov_features_predictions/model_predictions.csv')

    # Read in the data
    df_simple = pd.read_csv(pred_file_simple)
    df_full = pd.read_csv(pred_file_full)
    print('len(df_simple):', len(df_simple))

    # Merge the dataframes on "id" and "truth" to keep only overlapping rows.
    df_merged = pd.merge(df_simple, df_full, on=["id", "truth"], suffixes=('_simple', '_full'))
    print('len(df_merged):', len(df_merged))

    # Create a set of composite keys (id, truth) from the merged DataFrame.
    merged_keys = set(zip(df_merged['id'], df_merged['truth']))

    # Filter rows based on the composite key.
    filtered_simple = df_simple[
        ~df_simple.apply(lambda row: (row['id'], row['truth']) in merged_keys, axis=1)
    ]
    filtered_full = df_full[
        ~df_full.apply(lambda row: (row['id'], row['truth']) in merged_keys, axis=1)
    ]

    # For clarity, if you need the rows that did merge:
    filtered_out_simple = df_simple[
        df_simple.apply(lambda row: (row['id'], row['truth']) in merged_keys, axis=1)
    ]
    filtered_out_full = df_full[
        df_full.apply(lambda row: (row['id'], row['truth']) in merged_keys, axis=1)
    ]

    print('len(filtered_out_simple):', len(filtered_out_simple))
    print('len(filtered_out_full):', len(filtered_out_full))

    # if not filtered_simple.empty:
    #     print("Filtered out from simple model predictions:")
    #     print(filtered_simple[["id", "truth"]])

    # if not filtered_full.empty:
    #     print("Filtered out from full model predictions:")
    #     print(filtered_full[["id", "truth"]])
    
    # Use the merged dataframe for further analysis.
    df_simple = df_merged.filter(regex='^(id|truth|.*_simple)$').rename(columns=lambda x: x.replace('_simple', ''))
    df_full = df_merged.filter(regex='^(id|truth|.*_full)$').rename(columns=lambda x: x.replace('_full', ''))
    
    print('len(df_simple)', len(df_simple))
    print('len(df_full)', len(df_full))
    # print('df_simple.keys()', df_simple.keys())
   
    # Check that both files have the same number of rows and the same sample IDs.
    if len(df_simple) != len(df_full):
        raise ValueError("The simple and full prediction files have different numbers of rows.")
    if not df_simple["id"].equals(df_full["id"]):
        raise ValueError("The sample IDs in the two prediction files do not match.")
    
    # Identify prediction probability columns (those starting with "pred_")
    pred_cols = [col for col in df_simple.columns if col.startswith("pred_")]
    if not pred_cols:
        raise ValueError("No prediction probability columns (starting with 'pred_') found in the simple predictions file.")
    
    # Compute predicted labels from the probability columns.
    preds_simple = np.argmax(df_simple[pred_cols].values, axis=1)
    preds_full   = np.argmax(df_full[pred_cols].values, axis=1)
    
    # Use df_simple as the base for the analysis (it includes truth and physical features).
    df_analysis = df_simple.copy()
    df_analysis["pred_simple"] = preds_simple
    df_analysis["pred_full"]   = preds_full
    
    if "truth" not in df_analysis.columns:
        raise ValueError("The predictions CSV must include a 'truth' column.")
    
    test_dir  = "JetClass_gromov_delta_results_EMD/test"
    test_files = glob.glob(os.path.join(test_dir, "*.csv"))

    physical_features = load_physics_features(test_files,label_map=label_map)
    
    print('len(physical_features):', len(physical_features))
    print(physical_features.head(10))
    # Filter the physical features to match the filtered rows above
    physical_features_filtered = physical_features[
        physical_features.apply(lambda row: (row['index'], row['label']) in merged_keys, axis=1)
    ]
    
    print('len(physical_features_filtered):', len(physical_features_filtered))

    # Merge the filtered physical features with the analysis DataFrame
    df_analysis = pd.merge(df_analysis, physical_features_filtered, left_on=["id", "truth"], right_on=["index", "label"], how="left")

    # Update with correct loading, not right needs to load from JetClass
    if physical_features is None:
        exclude_cols = {"id", "truth", "true_str", "features"}
        physical_features = [col for col in df_analysis.columns 
                             if (col not in exclude_cols and not col.startswith("pred_"))]
        print("Inferred physical feature columns:", physical_features)
    
    # Mark correctness of each model's prediction.
    df_analysis["correct_simple"] = (df_analysis["pred_simple"] == df_analysis["truth"])
    df_analysis["correct_full"]   = (df_analysis["pred_full"] == df_analysis["truth"])
    
    # Define group based on correctness.
    def group_label(row):
        if row["correct_simple"] and row["correct_full"]:
            return "both_correct"
        elif (not row["correct_simple"]) and (not row["correct_full"]):
            return "both_incorrect"
        elif row["correct_simple"] and (not row["correct_full"]):
            return "simple_correct"
        else:
            return "full_correct"
    
    df_analysis["group"] = df_analysis.apply(group_label, axis=1)
    
    # Save the combined analysis DataFrame.
    combined_csv = os.path.join(out_dir, "combined_analysis.csv")
    df_analysis.to_csv(combined_csv, index=False)
    print("Combined analysis CSV saved to:", combined_csv)
    
    # --------------------------
    # Create and Save Plots
    # --------------------------
    
    # # Plot 1: Count plot for the four groups.
    # plt.figure(figsize=(8,6), dpi = 100)
    # order = ["both_correct", "simple_correct", "full_correct", "both_incorrect"]
    # sns.countplot(data=df_analysis, x="group", order=order)
    # plt.xlabel("Group", fontsize=18)
    # plt.ylabel("Count", fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "group_counts.png"))
    # plt.close()
    
    # # Plot 2: Scatter plot of jet_pt vs. jet_eta, if available.
    # if "jet_pt" in physical_features and "jet_eta" in physical_features:
    #     plt.figure(figsize=(8,6), dpi = 100)
    #     sns.scatterplot(data=df_analysis, x="jet_pt", y="jet_eta", hue="group", style="group", alpha=0.7)
    #     plt.xlabel("jet_pt", fontsize=18)
    #     plt.ylabel("jet_eta", fontsize=18)
    #     plt.xticks(fontsize=16)
    #     plt.yticks(fontsize=16)
    #     plt.legend(title="Group", labels=["Both Correct", "Limited Correct", "Gromov Correct", "Both Incorrect"], fontsize=16, loc='upper right')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(out_dir, "scatter_jet_pt_jet_eta.png"))
    #     plt.close()
    
    # # Plot 3: Overlaid KDE histograms for each physical feature by group.
    # for feature in physical_features:
    #     plt.figure(figsize=(8,6), dpi = 100)
    #     for grp, label in zip(df_analysis["group"].unique(), ["Both Correct", "Limited Correct", "Gromov Correct", "Both Incorrect"]):
    #         subset = df_analysis[df_analysis["group"] == grp]
    #         sns.kdeplot(subset[feature], label=label, fill=True, common_norm=False, alpha=0.5)
    #     plt.xlabel(feature, fontsize=18)
    #     plt.ylabel("Density", fontsize=18)
    #     plt.xticks(fontsize=16)
    #     plt.yticks(fontsize=16)
    #     plt.legend(fontsize=16, loc='upper right')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(out_dir, f"hist_{feature}.png"))
    #     plt.close()
    
    # # Plot 4: Pairplot of physical features (sampled if necessary).
    # if len(df_analysis) > sample_for_pairplot:
    #     df_pair = df_analysis.sample(n=sample_for_pairplot, random_state=42)
    # else:
    #     df_pair = df_analysis.copy()
    # pairplot = sns.pairplot(df_pair, vars=physical_features, hue="group", diag_kind="kde", corner=True)
    # pairplot_file = os.path.join(out_dir, "pairplot_features.png")
    # pairplot.savefig(pairplot_file)
    # plt.close()
    
    # Plot 5: Correlation heatmap for each group using physical features.
    valid_physical_features = [col for col in physical_features if col in df_analysis.columns]
    group_means = df_analysis.groupby("group")[valid_physical_features].mean().T
    plt.figure(figsize=(10,8), dpi=100)
    sns.heatmap(group_means, annot=True, cmap="coolwarm", vmin=group_means.values.min(), vmax=group_means.values.max())
    plt.xlabel("Group", fontsize=18)
    plt.ylabel("Physical Features", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_vs_physical_features_heatmap.png"))
    plt.close()
    print("All analysis plots have been saved to:", out_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare model performance using predictions CSV files that include physics data."
    )
    parser.add_argument("--log_dir", required=True,
                        help="Path to log dir for the model.")
    parser.add_argument("--out_dir", default="analysis_plots",
                        help="Output directory to save analysis plots and CSV.")
    parser.add_argument("--sample_for_pairplot", type=int, default=1000,
                        help="Maximum number of samples to include in the pairplot.")
    parser.add_argument("--physical_features", nargs="+", default=None,
                        help="List of physical feature column names to use for analysis. "
                             "If not provided, the script will try to infer them.")
    args = parser.parse_args()
    
    analyze_model_performance(
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        sample_for_pairplot=args.sample_for_pairplot,
        physical_features=args.physical_features
    )

if __name__ == "__main__":
    main()
