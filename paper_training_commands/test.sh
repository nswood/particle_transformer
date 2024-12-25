#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu_test
#SBATCH --time=01:00:00

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

curvatureInit=2
PMweightinit=1
interManFreq=-1
num_epochs=30
clamp=-1
batchsize=128
lrsched='flat+decay'
conv_embed="False"
betas="0.9,0.999"
learning_rate=5e-4
dropout_rate=0.1
optimizer='r_adam'
weight_decay=0
decay_step=0.5
grad_accum=1
epochs=30

DDP_NGPUS=1 ./train_JetClass.sh PMTrans RxH 2 RxH 2 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'test' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 1 --network-option num_heads 1 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step}  --network-option pair_embed_dims None


dimensions=(16)  # Add your dimensions here

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans R "$dimension" R 16 kinpid --network-option dropout_rate 0.2
#     DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH "$half_dimension" h4q --dev-id 'PMNN_testing_h4q' --start-lr 1e-3 --num-epochs 25
#     DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R "$dimension" R 16 kin --num-epochs 20 --dev-id 'Euclidean_20_epochs'
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans H "$dimension" R 16 kinpid
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans S "$dimension" R 16 kinpid
    
    # Commands with dimensions / 2
    
    DDP_NGPUS=1 COMMENT=TEST ./train_JetClass.sh PMNN H "$dimension" h4q --dev-id 'PMNN_more_runs' --start-lr 1e-3 --num-epochs 1 --decay-steps 0.3 --optimizer 'r_adam' --PM-weight-initialization-factor 0.1 --skip-test
    
#     DDP_NGPUS=1 COMMENT=TEST ./train_JetClass.sh PMTrans RxH 120 RxH 128 kinpid --PM-weight-initialization-factor 10 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'comparison_v3' --att-metric 'tan_space' --num-epochs 10 --base-resid-agg --base-activations 'act' --network-option  conv_embed 'True' --network-option  clamp 'True' --optimizer 'riemann_ranger' --grad-accum 4 --decay-steps 0.5
    
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans RxS "$half_dimension" R 16 kinpid
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans HxS "$half_dimension" R 16 kinpid
#     DDP_NGPUS=1 ./train_JetClass.sh PMTrans RxR "$half_dimension" R 16 kinpid
    
done
