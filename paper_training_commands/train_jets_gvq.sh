#!/bin/bash
#SBATCH --job-name=qg_jets
#SBATCH --partition=gpu
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
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


dimensions=(8 16 32 64)

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
    DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans R 64 R "$dimension"
    DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans R 64 H "$dimension"
    DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans R 64 S "$dimension"
    
    # Commands with dimensions / 2
    DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans R 64 RxH "$half_dimension"
    DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans R 64 RxS "$half_dimension"
    DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans R 64 HxS "$half_dimension"
    DDP_NGPUS=2 ./train_QuarkGluon.sh PMTrans R 64 RxR "$half_dimension" 
    
done


# # 4D
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 R 4
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 H 4
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 S 4
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxH 2
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 HxS 2
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxS 2


# # 8D
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 R 8
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxH 4
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxS 4
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 HxS 4
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 S 8
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 H 8

# # 16D
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 R 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxH 8
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxS 8
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 HxS 8
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 S 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 H 16

# # # 32D
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 R 32
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxH 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxS 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 HxS 16
# # DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 S 32
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 H 32

# 64D
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 R 128 R 128
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 RxS 32 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 HxS 32 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 S 64 R 16
# DDP_NGPUS=4 ./train_QuarkGluon.sh PMTrans R 32 H 64 R 16



