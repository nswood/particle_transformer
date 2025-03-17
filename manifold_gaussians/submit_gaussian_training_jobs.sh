#!/bin/bash


part_dims=(2 4 8 16)
lrs=(0.01 0.001 0.0001)
batch_sizes=(512 1024 2048)

outdir_base="gaussian_classification_final_model_higher_reg"
for lr in "${lrs[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for part_dim in "${part_dims[@]}"; do
            JOB_NAME="gaussian_classification_R_${part_dim}_${lr}_${batch_size}"

            CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R --part_dim ${part_dim} --part_curvature_init '0' --batch_size ${batch_size} --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr ${lr} --epochs 100 --model_name R_${part_dim}_${lr}_${batch_size} --outdir ${outdir_base} --visualize_outputs" 
            sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

            CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim ${part_dim} --part_curvature_init 'm1' --batch_size ${batch_size} --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr ${lr} --epochs 100 --model_name H_${part_dim}_${lr}_${batch_size} --outdir ${outdir_base} --visualize_outputs" 
            sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

            CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim ${part_dim} --part_curvature_init '1' --batch_size ${batch_size} --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr ${lr} --epochs 100 --model_name S_${part_dim}_${lr}_${batch_size} --outdir ${outdir_base} --visualize_outputs" 
            sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
            
            # #-----------------------------------
            
            if [ "${part_dim}" -lt 16 ]; then
                CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,R --part_dim ${part_dim} --part_curvature_init '0,0' --batch_size ${batch_size} --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr ${lr} --epochs 100 --model_name RxR_${part_dim}_${lr}_${batch_size} --outdir ${outdir_base} --visualize_outputs" 
                sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
        
                CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,M --part_dim ${part_dim} --part_curvature_init '0,m1' --batch_size ${batch_size} --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr ${lr} --epochs 100 --model_name RxH_${part_dim}_${lr}_${batch_size} --outdir ${outdir_base} --visualize_outputs" 
                sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
        
                CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,R --part_dim ${part_dim} --part_curvature_init '1,0' --batch_size ${batch_size} --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr ${lr} --epochs 100 --model_name SxR_${part_dim}_${lr}_${batch_size} --outdir ${outdir_base} --visualize_outputs" 
                sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
        
                CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,M --part_dim ${part_dim} --part_curvature_init '1,m1' --batch_size ${batch_size} --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr ${lr} --epochs 100 --model_name SxH_${part_dim}_${lr}_${batch_size} --outdir ${outdir_base} --visualize_outputs" 
                sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
            fi

            
        done
    done
done
echo "All jobs submitted."

