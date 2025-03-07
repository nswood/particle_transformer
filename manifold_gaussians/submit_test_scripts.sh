#!/bin/bash
#SBATCH --job-name=testing         # Job name from the first argument
#SBATCH --partition=gpu_test  # Use the GPU partition
#SBATCH --time=4:00:00        # Set a 12-hour time limit
#SBATCH --nodes=1              # Single node
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --ntasks-per-node=1    # Total of 4 tasks (GPUs) per node# Request 4 GPUs per node
#SBATCH --cpus-per-task=1      # Set CPUs per task
#SBATCH --mem=150G             # Set memory per node
#SBATCH --chdir=/n/home11/nswood/particle_transformer/
#SBATCH --output=slurm_test_toy_gaussians/%x-%j.out  # Standard output file

# Activate the environment and source the required setup
source ~/.bashrc
source /n/holystore01/LABS/iaifi_lab/Users/nswood/mambaforge/etc/profile.d/conda.sh 
conda activate top_env
    
# Run the command provided as an argument
python manifold_gaussians/train_gaussian_classification.py --part_geom R --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name R_2 --outdir gaussians_first_comparison

python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name H_2 --outdir gaussians_first_comparison

python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name H_2_map_off --map_off_manifold True --outdir gaussians_first_comparison

python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name S_2 --outdir gaussians_first_comparison

python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name S_2_map_off --map_off_manifold True --outdir gaussians_first_comparison

python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name H_2_learnable --outdir gaussians_first_comparison --part_curvature_trainable True

python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name H_2_map_off_learnable --map_off_manifold True --outdir gaussians_first_comparison  --part_curvature_trainable True


python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name S_2_learnable --outdir gaussians_first_comparison  --part_curvature_trainable True


python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --lr 0.0001 --epochs 200 --model_name S_2_map_off_learnable --map_off_manifold True --outdir gaussians_first_comparison  --part_curvature_trainable True

