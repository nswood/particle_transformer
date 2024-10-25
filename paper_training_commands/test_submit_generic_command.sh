#!/bin/bash
#SBATCH --job-name=PM_job        # Job name from the first argument
#SBATCH --partition=gpu_test        # Use the GPU partition
#SBATCH --time=01:00:00        # Set a 12-hour time limit
#SBATCH --nodes=1              # Single node
#SBATCH --ntasks-per-node=1   # Total of 4 tasks (GPUs) per node
#SBATCH --gres=gpu:1           # Request 4 GPUs per node
#SBATCH --cpus-per-task=1      # Set CPUs per task
#SBATCH --mem=50G             # Set memory per node
#SBATCH --chdir=/n/home11/nswood/particle_transformer/
#SBATCH --output=slurm_monitoring/%x-%j.out  # Standard output file



# Activate the environment and source the required setup
source ~/.bashrc
source /n/holystore01/LABS/iaifi_lab/Users/nswood/mambaforge/etc/profile.d/conda.sh
conda activate top_env

# Run the command provided as an argument
COMMAND_STRING="$2"
# echo "SLURM ID: $1"
# echo "Executing command: $COMMAND_STRING"
eval $COMMAND_STRING
