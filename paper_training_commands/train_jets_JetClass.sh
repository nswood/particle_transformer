#!/bin/bash
#SBATCH --job-name=jc_j
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=450G
#SBATCH --chdir=/n/home11/nswood/particle_transformer/
#SBATCH --output=slurm_monitoring/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=18301
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

dimensions=(16 32 48 64)



# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
    DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 R "$dimension" kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'
    DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 H "$dimension" kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'
    DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 S "$dimension" kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'

    
#     # Commands with dimensions / 2
    DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 RxH "$half_dimension" kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'
    DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 RxS "$half_dimension" kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'
    DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 HxS "$half_dimension" kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'

    
done




