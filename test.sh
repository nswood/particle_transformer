#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu_test
#SBATCH --time=08:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==1)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --chdir=/n/home11/nswood/particle_transformer/
#SBATCH --output=slurm_monitoring/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=19304
export WORLD_SIZE=1

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


# DDP_NGPUS=1 ./train_CLR_JetClass.sh PMTrans R 4 R 16 kinpid

# DDP_NGPUS=1 ./train_JetClass.sh PMTrans R 32 R 16 kinpid


dimensions=(256 )  # Add your dimensions here

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans R "$dimension" R 16 kinpid
#     DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R "$dimension" R 16 kin --num-epochs 20 --dev-id 'Euclidean_20_epochs'
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans H "$dimension" R 16 kinpid
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans S "$dimension" R 16 kinpid
    
    # Commands with dimensions / 2
    DDP_NGPUS=1 ./train_JetClass.sh PMTrans RxH "$half_dimension" R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 2 --inter-man-att-method 'v2' --dev-id 'tan_space_att_metric' --att-metric 'tan_space' 
    
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans RxS "$half_dimension" R 16 kinpid
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans HxS "$half_dimension" R 16 kinpid
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans RxR "$half_dimension" R 16 kinpid
    
done
