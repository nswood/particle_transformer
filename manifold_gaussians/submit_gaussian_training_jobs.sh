#!/bin/bash
part_dims=(2 4 8)
repeat_trainings=3
outdir_base="guassian_toys_initial_tests"
for part_dim in "${part_dims[@]}"; do
    for ((i=1; i<=repeat_trainings; i++)); do
        JOB_NAME="gaussian_clustering_dim_${part_dim}_run_${i}"
        CMD="python manifold_gaussians/train_gaussian_clustering.py --part_geom R --part_dim ${part_dim} --epochs 250 --part_curvature_init -1 --batch_size 64 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --model_name R_${part_dim} --outdir ${outdir_base}"
        sbatch manifold_gaussians/submit_command.sh ${JOB_NAME} "$CMD"
    
        CMD="python manifold_gaussians/train_gaussian_clustering.py --part_geom M --part_dim ${part_dim} --epochs 250 --part_curvature_init -1 --batch_size 64 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --model_name H_${part_dim} --outdir ${outdir_base}"
        sbatch manifold_gaussians/submit_command.sh ${JOB_NAME} "$CMD"
    
        CMD="python manifold_gaussians/train_gaussian_clustering.py --part_geom M --part_dim ${part_dim} --epochs 250 --part_curvature_init 1 --batch_size 64 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy --model_name S_${part_dim} --outdir ${outdir_base}"
        sbatch manifold_gaussians/submit_command.sh ${JOB_NAME} "$CMD"
    done
done


echo "All jobs submitted."
