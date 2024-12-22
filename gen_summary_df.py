import os
import uproot
import awkward as ak
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score


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

# Function to perform bootstrapping and calculate rejection uncertainty
def bootstrap_rejections(y_true, y_score, target_tprs, samples, sample_size):
    bootstrap_rejections = {tpr: [] for tpr in target_tprs}
    
    for _ in range(samples):
        indices = np.random.choice(range(len(y_true)), size=sample_size, replace=True)
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


def bootstrap_acc(y_true, y_pred, samples, sample_size):
    bootstrap_accuracies = []
    
    for _ in range(samples):
        # Randomly select indices for the bootstrap sample
        indices = np.random.choice(range(len(y_true)), size=sample_size, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
                
        # Calculate accuracy: proportion of correct predictions
        accuracy = np.mean(y_pred_sample == y_true_sample)
        bootstrap_accuracies.append(accuracy)
    
    # Calculate the mean and standard deviation of the bootstrapped accuracies
    acc_mean = np.mean(bootstrap_accuracies)
    acc_std = np.std(bootstrap_accuracies)
    
    return acc_mean, acc_std


# Main function to process all files and save metrics incrementally
def process_directory(directory_path, csv_file_path, samples, sample_size):
    first_write = not os.path.exists(csv_file_path)  # Check if the CSV file already exists
    trained_models = []
    if not first_write:
        trained_models = pd.read_csv(csv_file_path)['model_id'].tolist()
        print(trained_models)
        
    if 'h4q' in directory_path:
        signals = [ 'H4q', 'QCD']
        PMNN = True
    elif 'tbqq' in directory_path:
        signals = ['Tbqq', 'QCD']
        PMNN = True
    else:
        signals = ['Tbqq', 'Tbl', 'Hqql', 'Wqq', 'Hgg', 'H4q', 'Hbb', 'QCD', 'Hcc', 'Zqq']
        PMNN = False
        
        
    # Outer loop: iterate over subdirectories (each representing a model)
    for model_name in os.listdir(directory_path):
        print(f'Loading {model_name}')
        try:
            if model_name not in trained_models:
                
                model_path = os.path.join(directory_path, model_name, 'predict_output')
                if os.path.isdir(model_path):
                    print(f"Processing model: {model_name}")
                    model_data = pd.DataFrame()  # DataFrame to hold combined data for this model

                    # Inner loop: iterate over .root files in the model's subdirectory
                    for root_file in os.listdir(model_path):
                        if root_file.endswith('.root'):
                            file_path = os.path.join(model_path, root_file)
                            df = read_root_file(file_path)  # Read the root file
                            model_data = pd.concat([model_data, df], axis=0)

                    # Extract score and label columns in the order of signals list
                    score_cols = [f'score_label_{signal}' for signal in signals]
                    label_cols = [f'label_{signal}' for signal in signals]
                    # Ensure that only the relevant columns exist in model_data
                    score_cols = [col for col in score_cols if col in model_data.columns]
                    label_cols = [col for col in label_cols if col in model_data.columns]

                    # Use DataFrame values to get predictions directly
                    preds = model_data[score_cols].values.tolist()
                    # One-hot encode true labels
                    true_label_indices = model_data[label_cols].values.argmax(axis=1)
                    labels = np.eye(len(signals))[true_label_indices].tolist()
                    print(len(labels))
                    # Calculate overall accuracy
#                     if PMNN:
#                         overall_accuracy = accuracy_score(np.argmax(labels, axis=1), 1-np.round(preds))
#                     else:
                    overall_accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(preds, axis=1))
                        
                    print(f"Overall accuracy for model {model_name}: {overall_accuracy}")

                    if overall_accuracy < ((1/len(signals))+0.05):
                        continue
                    metrics = {
                        'model_id': model_name,
                        'overall_accuracy': overall_accuracy,
                        'base_name': ''.join(model_name.split('_batch1024')[1:])
                    }

                    if PMNN:
                        acc_mean, acc_std =  bootstrap_acc(np.argmax(labels, axis=1), np.argmax(preds, axis=1),samples, sample_size)
                    metrics['acc_mean'] = acc_mean
                    metrics['acc_std'] = acc_std
                    
                    # Initialize a dictionary to hold metrics for each class
                   
                    # Calculate metrics for each class
                    for i, signal in enumerate(signals):
                        
                        
                        y_true = np.array([label[i] for label in labels])
                        y_score = np.array([pred[i] for pred in preds])
                        
                        auc, accuracy, rejections, _, _ = calculate_metrics(y_true, y_score)

                        # Bootstrapping for rejection rate uncertainty
                        rejection_stats = bootstrap_rejections(y_true, y_score, [0.3, 0.5, 0.7, 0.99], samples, sample_size)

                        metrics.update({
                            f'{signal}_auc': auc,
                            f'{signal}_accuracy': accuracy,
                            f'{signal}_rejection_30_mean': rejection_stats[0.3][0],
                            f'{signal}_rejection_30_std': rejection_stats[0.3][1],
                            f'{signal}_rejection_50_mean': rejection_stats[0.5][0],
                            f'{signal}_rejection_50_std': rejection_stats[0.5][1],
                            f'{signal}_rejection_70_mean': rejection_stats[0.7][0],
                            f'{signal}_rejection_70_std': rejection_stats[0.7][1],
                            f'{signal}_rejection_99_mean': rejection_stats[0.99][0],
                            f'{signal}_rejection_99_std': rejection_stats[0.99][1],
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
    parser.add_argument('--sample_size', type=int, required=True, help="Size of each bootstrap sample.")
    
    # Parse the arguments
    args = parser.parse_args()

    # Specify the CSV file path using the provided argument
    csv_file_path = f'{args.opath}.csv'

    # Process the directory and save results incrementally
    process_directory(directory_path=args.dir_path, csv_file_path=csv_file_path, samples=args.samples, sample_size=args.sample_size)

    print(f"The results have been saved incrementally to {csv_file_path}")

if __name__ == '__main__':
    main()
