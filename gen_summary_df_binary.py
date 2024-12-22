import os
import uproot
import awkward as ak
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

signals = ['QCD','Hbb','Hcc','Hgg','H4q','Hqql','Zqq','Wqq', 'Tbqq', 'Tbl']

# Function to read data from ROOT file and convert to a Pandas DataFrame
def read_root_file(file_path):
    with uproot.open(file_path) as file:
        # Assuming the tree name is the first key in the file
        tree = file[file.keys()[0]]
        # Convert the tree to an awkward array
        data = tree.arrays(library='ak')
    # Convert the awkward array to an Arrow table, then to a Pandas DataFrame
    arrow_table = ak.to_arrow_table(data)
    return arrow_table.to_pandas()

# Function to calculate metrics for each class
def calculate_metrics(y_true, y_score, target_tprs=[0.3, 0.5, 0.7, 0.99]):
    if len(np.unique(y_true)) < 2:
        return None, None, {tpr: None for tpr in target_tprs}, None, None
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_score >= 0.5)
    
    rejections = {}
    for tpr_value in target_tprs:
        try:
            target_index = np.where(tpr >= tpr_value)[0][0]
            target_fpr = fpr[target_index]
            rejections[tpr_value] = 1 / target_fpr if target_fpr != 0 else np.inf
        except IndexError:
            rejections[tpr_value] = None
    
    return auc, accuracy, rejections, fpr.tolist(), tpr.tolist()


# Function to perform bootstrapping and calculate accuracy and AUC with 'macro' averaging and 'ovo' multi-class
def bootstrap_accuracy_auc(preds, labels, samples, sample_percent):
    # Containers for bootstrap statistics
    bootstrap_accuracies = []
    bootstrap_aucs = []
    
    for i in range(samples):
        # Sample with replacement from indices
        indices = np.random.choice(range(len(labels)), size=int(sample_percent*len(labels)), replace=True)
        labels_sample = labels[indices]
        preds_sample = preds[indices]
        n_classes = labels_sample.shape[1]
        
        # Calculate accuracy
        y_true_labels = np.argmax(labels_sample, axis=1)
        y_pred_labels = np.argmax(preds_sample, axis=1)
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        bootstrap_accuracies.append(accuracy)
        
        # Calculate AUC for each class using one-vs-one (ovo) approach and macro average
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true_labels, preds_sample[:,1])
            else:
                auc = roc_auc_score(y_true_labels, preds_sample, average='macro', multi_class='ovo')
            
            bootstrap_aucs.append(auc)
        except ValueError as e:
            print(f"Error in AUC calculation for sample {i}: {e}")
            # Handle cases where AUC calculation fails (e.g., not enough samples per class)
            bootstrap_aucs.append(None)
    
    # Calculate statistics
    accuracy_mean = np.mean(bootstrap_accuracies)
    accuracy_std = np.std(bootstrap_accuracies)
    auc_mean = np.mean([auc for auc in bootstrap_aucs if auc is not None])
    auc_std = np.std([auc for auc in bootstrap_aucs if auc is not None])
#     print(auc_std)
    
    return {
        'accuracy': (accuracy_mean, accuracy_std),
        'auc': (auc_mean, auc_std)
    }

# Function to perform bootstrapping and calculate rejection uncertainty
def bootstrap_rejections(y_true, y_score, target_tprs, samples, sample_percent):
    
    bootstrap_rejections = {tpr: [] for tpr in target_tprs}
    
    for _ in range(samples):
        indices = np.random.choice(range(len(y_true)), size=int(sample_percent*len(y_true)), replace=True)
        y_true_sample = y_true[indices]
        y_score_sample = y_score[indices]
        
        _, _, rejections, _, _ = calculate_metrics(y_true_sample, y_score_sample, target_tprs)
        
        
        
        for tpr in target_tprs:
            if rejections[tpr] is not None:
                bootstrap_rejections[tpr].append(rejections[tpr])
        
    
    rejection_stats = {}
    for tpr in target_tprs:
        if bootstrap_rejections[tpr]:
            rejection_mean = np.mean(bootstrap_rejections[tpr])
            rejection_std = np.std(bootstrap_rejections[tpr])
            rejection_stats[tpr] = (rejection_mean, rejection_std)
        else:
            rejection_stats[tpr] = (None, None)
    
    
    return rejection_stats

# Main function to process all files and save metrics incrementally
def process_directory(directory_path, csv_file_path, samples, sample_percent, name = 'top', pt_percent = -1):
    first_write = not os.path.exists(csv_file_path)  # Check if the CSV file already exists
    if not first_write:
        trained_models = pd.read_csv(csv_file_path)['model_id'].tolist()
        print(trained_models)
    else:
        trained_models = []
    target_tprs=[0.3, 0.5, 0.7, 0.99]
    # Outer loop: iterate over subdirectories (each representing a model)
    for model_name in os.listdir(directory_path):
        print(f'Loading {model_name}')
        try:
            if model_name not in trained_models:
                model_path = os.path.join(directory_path, model_name, 'predict_output')
        #         print(model_name)
                if os.path.isdir(model_path):
                    model_data = pd.DataFrame()  # DataFrame to hold combined data for this model
                    preds = []  # List to store predictions
                    labels = []  # List to store one-hot encoded labels

        #             
                    # Inner loop: iterate over .root files in the model's subdirectory
                    for root_file in os.listdir(model_path):
                        if root_file.endswith('.root'):
                            file_path = os.path.join(model_path, root_file)
                            df = read_root_file(file_path) # Read the root file
                            model_data = pd.concat([model_data, df], axis=0)
                    if pt_percent > 0:
                        threshold = model_data['jet_pt'].quantile(pt_percent)
                        # Select rows where jet_pt is greater than or equal to the threshold
                        model_data = model_data[model_data['jet_pt'] >= threshold]

                    score_cols = [col for col in model_data.columns if col.startswith('score_')]
                    if 'Top' in directory_path:
                        label_cols = ['jet_isQCD','jet_isTop']
                        score_cols = ['score_jet_isQCD','score_jet_isTop']
                        signals = ['QCD','Top']

                        n_signals =2
                        
                    elif 'QuarkGluon' in directory_path:
                        label_cols = ['jet_isQ','jet_isG']
                        score_cols = ['score_jet_isQ','score_jet_isG']
                        label_cols = [col for col in model_data.columns if col.startswith('jet_is')]
                        signals = ['Quark','Gluon']

                        n_signals =2
                    elif 'PMNN' in directory_path:
                        label_cols = [col for col in model_data.columns if col.startswith('label_')]
                        n_signals =2
                        if 'h4q' in directory_path:
                            signals = ['QCD','h4q']
                        else:
                            signals = ['QCD','tbqq']
                    else:
                        label_cols = [col for col in model_data.columns if col.startswith('label_')]
                        signals = ['QCD','Hbb','Hcc','Hgg','H4q','Hqql','Zqq','Wqq', 'Tbqq', 'Tbl']
                        n_signals = len(signals)


                    # Use DataFrame values to get predictions directly
                    preds = model_data[score_cols].values.tolist()
                    
#                     print(model_data.columns)
                    

                    # One-hot encode true labels
                    true_label_indices = model_data[label_cols].values.argmax(axis=1)
                    labels = np.eye(n_signals)[true_label_indices].tolist()
                    
                    # Calculate overall accuracy
                    overall_accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(preds, axis=1))
                    overall_auc = roc_auc_score(np.argmax(labels, axis=1), np.array(preds)[:,1])
                    # Now preds contains all the predictions and labels contains all the one-hot encoded true labels
                    # Convert lists to DataFrame
                    
                    metrics = {
                        'model_id': model_name,
                        'overall_accuracy': overall_accuracy,
                        'overall_auc': overall_auc,
                        'base_name': ''.join(model_name.split('_batch128')[1:])
                    }
                    
                    
                    
                    rejection_stats = bootstrap_accuracy_auc(np.array(labels), np.array(preds), samples, sample_percent)
                    metrics.update({
                            f'auc_mean':rejection_stats['auc'][0],
                            f'auc_std':rejection_stats['auc'][1],
                            f'accuracy_mean':rejection_stats['accuracy'][0],
                            f'accuracy_std':rejection_stats['accuracy'][1],
                    })
                    

                    # Calculate metrics for each class
                    for i in range(1,n_signals):
                        if n_signals == len(signals):
                            name = signals[i]
                        y_true = np.array([label[i] for label in labels])
                        y_score = np.array([(pred[i])/(pred[i] + pred[0] + 10e-10) for pred in preds])

                        auc, accuracy, rejections, _, _ = calculate_metrics(y_true, y_score,target_tprs)

                        # Bootstrapping for rejection rate uncertainty
                        rejection_stats = bootstrap_rejections(y_true, y_score, [0.3, 0.5, 0.7, 0.99], samples, sample_percent)
                        metrics.update({
                            f'{name}_rejection_30_mean': rejection_stats[0.3][0],
                            f'{name}_rejection_30_std': rejection_stats[0.3][1],
                            f'{name}_rejection_50_mean': rejection_stats[0.5][0],
                            f'{name}_rejection_50_std': rejection_stats[0.5][1],
                            f'{name}_rejection_70_mean': rejection_stats[0.7][0],
                            f'{name}_rejection_70_std': rejection_stats[0.7][1],
                            f'{name}_rejection_99_mean': rejection_stats[0.99][0],
                            f'{name}_rejection_99_std': rejection_stats[0.99][1],
                        })
                        
                        # Full test data
                        _, _, rejections, _, _ = calculate_metrics(y_true, y_score, target_tprs)
                        for tpr in target_tprs:
                            if rejections[tpr] is not None:
#                                 bootstrap_rejections[tpr].append(rejections[tpr])
                                metrics.update({
                                    f'{name}_rejection_30_full': rejections[0.3],
                                    f'{name}_rejection_50_full': rejections[0.5],
                                    f'{name}_rejection_70_full': rejections[0.7],
                                    f'{name}_rejection_99_full': rejections[0.99],
                                })
                        

                    # Convert the metrics dictionary to a DataFrame
                    results_df = pd.DataFrame([metrics])
                    # Save the DataFrame to CSV, appending new results each time
                    results_df.to_csv(csv_file_path, mode='a', index=False, header=first_write)
                    trained_models.append(model_name)
                    first_write = False  # Only write header for the first model
        except OSError as e:
            print(f"OSError encountered while reading the file {file_path}: {e}")
            # Log the error or handle it in a suitable manner
            continue  # Skip to the next file if an OSError is encountered


    print(f"All results have been saved incrementally to {csv_file_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and save model results.")
    parser.add_argument('--dir_path', type=str, required=True, help="Path to the directory containing model data.")
    parser.add_argument('--opath', type=str, required=True, help="Output file path without extension for saving results.")
    parser.add_argument('--samples', type=int, required=True, help="Number of bootstrap samples to use.")
    parser.add_argument('--sample_percent', type=float, required=True, help="Percent of total dataset for each bootstrap sample.")
#     parser.add_argument('--name', type=str, required=True, help="Process name.")

    # Parse the arguments
    args = parser.parse_args()

    # Specify the CSV file path using the provided argument
    csv_file_path = f'{args.opath}.csv'

#     # Process the directory and save results incrementally
    process_directory(directory_path=args.dir_path, csv_file_path=csv_file_path, samples=args.samples, sample_percent=args.sample_percent)

    csv_file_path = f'{args.opath}_top_20.csv'

    # Process the directory and save results incrementally
    process_directory(directory_path=args.dir_path, csv_file_path=csv_file_path, samples=args.samples, sample_percent=args.sample_percent, pt_percent = 0.8)
    
    csv_file_path = f'{args.opath}_top_10.csv'

    # Process the directory and save results incrementally
    process_directory(directory_path=args.dir_path, csv_file_path=csv_file_path, samples=args.samples, sample_percent=args.sample_percent, pt_percent = 0.9)
    
    print(f"The results have been saved incrementally to {csv_file_path}")

if __name__ == '__main__':
    main()
