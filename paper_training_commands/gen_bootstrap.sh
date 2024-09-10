#!/bin/bash
#SBATCH --job-name=boot
#SBATCH --partition=test
#SBATCH --time=08:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --chdir=/n/home11/nswood/particle_transformer/
#SBATCH --output=slurm_monitoring/%x-%j.out


### init virtual environment if needed

# cd Mixed_Curvature
source ~/.bashrc

source /n/holystore01/LABS/iaifi_lab/Users/nswood/mambaforge/etc/profile.d/conda.sh

# conda activate flat-samples

conda activate top_env

# python gen_summary_df.py --dir_path /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/full/PMTrans --opath JetClass_processed_performance --samples 25 --sample_size 1000000

python gen_summary_df_binary.py --dir_path '/n/holystore01/LABS/iaifi_lab/Lab/nswood/training/TopLandscape/PMTrans' --opath TopLandscape_processed_performance --samples 25 --sample_size 200000 --name top

python gen_summary_df_binary.py --dir_path '/n/holystore01/LABS/iaifi_lab/Lab/nswood/training/QuarkGluon/PMTrans' --opath QuarkGluon_processed_performance --samples 25 --sample_size 100000 --name quarkgluon


# python gen_summary_df.py --dir_path /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/TopLandscape/PMTrans --opath JetClass_processed_performance --samples 25 --sample_size 1000000