#!/bin/bash

# part_dims=(2)
# curvature_init=(0.25)
# repeat_trainings=1
# outdir_base="gaussian_classification_learnable_curvature_test"


part_dims=(2 4)
curvature_init=(0.25 1 4)
repeat_trainings=3
outdir_base="gaussian_classification_learnable_curvature_lower_lr"

for ((i=1; i<=repeat_trainings; i++)); do
    for curv in "${curvature_init[@]}"; do
        for part_dim in "${part_dims[@]}"; do
            JOB_NAME="gaussian_classification_R_${part_dim}"

            # CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim ${part_dim} --part_curvature_init 'm${curv}' --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.001 --epochs 100 --model_name H_${part_dim}_${curv} --outdir ${outdir_base} --part_curvature_trainable"
            # sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

            # CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim ${part_dim} --part_curvature_init '${curv}' --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.001 --epochs 100 --model_name S_${part_dim}_${curv} --outdir ${outdir_base} --part_curvature_trainable"
            # sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
            
            # ##-----------------------------------

            # CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,M --part_dim ${part_dim} --part_curvature_init '0,m${curv}' --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.001 --epochs 100 --model_name RxH_${part_dim}_${curv} --outdir ${outdir_base} --part_curvature_trainable"
            # sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

            # CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,R --part_dim ${part_dim} --part_curvature_init '${curv},0' --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.001 --epochs 100 --model_name SxR_${part_dim}_${curv} --outdir ${outdir_base} --part_curvature_trainable"
            # sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

            # CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,M --part_dim ${part_dim} --part_curvature_init '${curv},m${curv}' --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.001 --epochs 100 --model_name SxH_${part_dim}_${curv} --outdir ${outdir_base} --part_curvature_trainable"
            # sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

            CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,M --part_dim ${part_dim} --part_curvature_init 'm${curv},m${curv}' --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.001 --epochs 100 --model_name HxH_${part_dim}_${curv} --outdir ${outdir_base} --part_curvature_trainable"
            sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

            CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,M --part_dim ${part_dim} --part_curvature_init '${curv},${curv}' --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.001 --epochs 100 --model_name SxS_${part_dim}_${curv} --outdir ${outdir_base} --part_curvature_trainable"
            sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
        done
    done
done

echo "All jobs submitted."

