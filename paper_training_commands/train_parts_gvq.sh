#!/bin/bash
#SBATCH --job-name=gvq_p
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --chdir=/n/home11/nswood/particle_transformer/
#SBATCH --output=slurm_monitoring/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=19304
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed

# cd Mixed_Curvature
source ~/.bashrc

source /n/holystore01/LABS/iaifi_lab/Users/nswood/mambaforge/etc/profile.d/conda.sh

# conda activate flat-samples

conda activate top_env

dimensions=(128 256)

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
#     DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans R "$dimension" R 16 kinpid --num-epochs 20 --PM-weight-initialization-factor 1 --inter-man-att 2 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'
#     DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans H "$dimension" R 16
#     DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans S "$dimension" R 16
    
    # Commands with dimensions / 2
    DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxH "$half_dimension" R 64 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act' --network-option num_layers 8
#     DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxS "$half_dimension" R 16
#     DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans HxS "$half_dimension" R 16
#     DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxR "$half_dimension" R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 2 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'
    
done

# # 4D
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 4 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxH 2 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxS 2 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans HxS 2 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans S 4 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans H 4 R 16

# # 8D
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 8 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxH 4 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxS 4 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans HxS 4 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans S 8 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans H 8 R 16

# # 16D
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 16 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxH 8 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxS 8 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans HxS 8 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans S 16 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans H 16 R 16

# # 32D
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxH 16 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxS 16 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans HxS 16 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans S 32 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans H 32 R 16

# # 64D
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 128 R 128
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans RxS 32 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans HxS 32 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans S 64 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans H 64 R 16



