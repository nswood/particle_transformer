#!/bin/bash
#SBATCH --job-name=jc_p
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

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
export MASTER_PORT=19322
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


dimensions=(96 64 48 32)

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
    DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R "$dimension" R 16 kinpid --num-epochs 20 --dev-id 'Euclidean_20_epochs' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'
    

done




dimensions=(128 96 64 48 32)

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH "$half_dimension" R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'
    
    DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS "$half_dimension" R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'

done


dimensions=(128)

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
    DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R "$dimension" R 16 kinpid --num-epochs 30 --dev-id 'Euclidean_30_epochs_paper' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'


done


dimensions=(160)

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
    DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH "$half_dimension" R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'

    DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS "$half_dimension" R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'

done
