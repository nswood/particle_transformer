#!/bin/bash
# repeat_trainings=3


# outdir_base="gaussians_updated_CL"

# JOB_NAME="gaussian_classification_R_2"
# CMD="python manifold_gaussians/train_gaussian_CL.py --part_geom R --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.0001 --epochs 50 --model_name R_2 --outdir gaussians_clr_updated"
# sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

# JOB_NAME="gaussian_classification_H_2_map_off"
# CMD="python manifold_gaussians/train_gaussian_CL.py --part_geom M --part_dim 2 --part_curvature_init -1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.0001 --epochs 50 --model_name H_2 --outdir gaussians_clr_updated"
# sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

# JOB_NAME="gaussian_classification_S_2"
# CMD="python manifold_gaussians/train_gaussian_CL.py --part_geom M --part_dim 2 --part_curvature_init 1 --batch_size 256 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.0001 --epochs 50 --model_name S_2 --outdir gaussians_clr_updated"
# sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"



# part_dims=(2 4 8)
# part_dims=(2 3 4 5 6 7 8)
# repeat_trainings=4
# outdir_base="gaussians_classification_adaptive_lr"
# for ((i=1; i<=repeat_trainings; i++)); do
#     for part_dim in "${part_dims[@]}"; do
#         JOB_NAME="gaussian_classification_R_${part_dim}"
#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R --part_dim ${part_dim} --part_curvature_init -1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name R_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"


#         JOB_NAME="gaussian_classification_H_${part_dim}"
#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim ${part_dim} --part_curvature_init -1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name H_${part_dim}  --outdir ${outdir_base} "
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

#         JOB_NAME="gaussian_classification_S_${part_dim}"
#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim ${part_dim} --part_curvature_init -1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name S_${part_dim}  --outdir ${outdir_base} "
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
#     done
# done


# part_dims=(2 3 4)
# repeat_trainings=5
# outdir_base="gaussians_classification_adaptive_lr"
# for ((i=1; i<=repeat_trainings; i++)); do
#     for part_dim in "${part_dims[@]}"; do
#         JOB_NAME="gaussian_classification_R_${part_dim}"
#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,R --part_dim ${part_dim} --part_curvature_init 0,0 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name RxR_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"


#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,M --part_dim ${part_dim} --part_curvature_init 0,-1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name RxH_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,M --part_dim ${part_dim} --part_curvature_init 0,1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name RxS_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,M --part_dim ${part_dim} --part_curvature_init -1,-1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name HxH_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,M --part_dim ${part_dim} --part_curvature_init 1,-1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name SxH_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

#         CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,M --part_dim ${part_dim} --part_curvature_init -1,-1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name SxS_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
#     done
# done

 

# part_dims=(2 4 8 16)
part_dims=(4 8 16)

repeat_trainings=5
outdir_base="gaussian_classification_pooling_comparison"
for ((i=1; i<=repeat_trainings; i++)); do
    for part_dim in "${part_dims[@]}"; do
        JOB_NAME="gaussian_classification_R_${part_dim}"


        # CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R --part_dim ${part_dim} --part_curvature_init 0 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name R_${part_dim} --outdir ${outdir_base}"
        # sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

        # CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim ${part_dim} --part_curvature_init -1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name H_${part_dim} --outdir ${outdir_base}"
        # sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

        # CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M --part_dim ${part_dim} --part_curvature_init 1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name S_${part_dim} --outdir ${outdir_base}"
        # sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
        
        # #-----------------------------------
        
        CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,R --part_dim ${part_dim} --part_curvature_init 0,0 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name RxR_${part_dim} --outdir ${outdir_base}"
        sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

        CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,M --part_dim ${part_dim} --part_curvature_init 0,-1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name RxH_${part_dim} --outdir ${outdir_base}"
        sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

        CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,R --part_dim ${part_dim} --part_curvature_init 1,0 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name SxR_${part_dim} --outdir ${outdir_base}"
        sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

        #-----------------------------------
        CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,R --part_dim ${part_dim} --part_curvature_init 0,0 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name RxR_${part_dim}_lgw --outdir ${outdir_base} --local_geom_weighting True"
        sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

        CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom R,M --part_dim ${part_dim} --part_curvature_init 0,-1 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name RxH_${part_dim}_lgw --outdir ${outdir_base} --local_geom_weighting True"
        sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"

        CMD="python manifold_gaussians/train_gaussian_classification.py --part_geom M,R --part_dim ${part_dim} --part_curvature_init 1,0 --batch_size 1024 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --lr 0.01 --epochs 100 --model_name SxR_${part_dim}_lgw --outdir ${outdir_base} --local_geom_weighting True"
        sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
    done
done





echo "All jobs submitted."






# for part_dim in "${part_dims[@]}"; do
#     for ((i=1; i<=repeat_trainings; i++)); do
#         JOB_NAME="gaussian_clustering_dim_${part_dim}_run_${i}"
#         CMD="python manifold_gaussians/train_gaussian_clustering.py --part_geom R --part_dim ${part_dim} --epochs 250 --part_curvature_init -1 --batch_size 64 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --model_name R_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
    
#         CMD="python manifold_gaussians/train_gaussian_clustering.py --part_geom M --part_dim ${part_dim} --epochs 250 --part_curvature_init -1 --batch_size 64 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --model_name H_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
    
#         CMD="python manifold_gaussians/train_gaussian_clustering.py --part_geom M --part_dim ${part_dim} --epochs 250 --part_curvature_init 1 --batch_size 64 --data_dir /n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated --model_name S_${part_dim} --outdir ${outdir_base}"
#         sbatch manifold_gaussians/submit_gpu_command.sh ${JOB_NAME} "$CMD"
#     done
# done