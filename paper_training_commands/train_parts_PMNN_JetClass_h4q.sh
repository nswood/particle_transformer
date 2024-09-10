#!/bin/bash
#SBATCH --job-name=h4q
#SBATCH --partition=gpu_test
#SBATCH --time=8:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==1)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
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

dimensions=(8 16 32 64)  # Add your dimensions here
# dimensions=(64 32)  # Add your dimensions here

for i in {1..2}; do
  echo "Iteration $i"
    # Loop over each dimension in the list
    for dimension in "${dimensions[@]}"; do
        half_dimension=$((dimension / 2))  # Calculate half of the dimension
        echo "Half-Dimension $half_dimension" 

        # Commands with dimensions / 2
#         DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH "$half_dimension" h4q
#         DDP_NGPUS=1 ./train_JetClass.sh PMNN RxS "$half_dimension" h4q
#         DDP_NGPUS=1 ./train_JetClass.sh PMNN HxS "$half_dimension" h4q
        DDP_NGPUS=1 ./train_JetClass.sh PMNN RxR "$half_dimension" h4q
        
        # Command with full dimension
#         DDP_NGPUS=1 ./train_JetClass.sh PMNN R "$dimension" h4q
        
#         DDP_NGPUS=1 ./train_JetClass.sh PMNN H "$dimension" h4q
#         DDP_NGPUS=1 ./train_JetClass.sh PMNN S "$dimension" h4q
        
    done
done

# dimensions=(64)  # Add your dimensions here

# for i in {1..10}; do
#   echo "Iteration $i"
#     # Loop over each dimension in the list
#     for dimension in "${dimensions[@]}"; do
#         half_dimension=$((dimension / 2))  # Calculate half of the dimension

#         # Commands with dimensions / 2
# #         DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH "$half_dimension" h4q
# #         DDP_NGPUS=1 ./train_JetClass.sh PMNN RxS "$half_dimension" h4q
# #         DDP_NGPUS=1 ./train_JetClass.sh PMNN HxS "$half_dimension" h4q
        
#         # Command with full dimension
# #         DDP_NGPUS=1 ./train_JetClass.sh PMNN R "$dimension" h4q
        
#         DDP_NGPUS=1 ./train_JetClass.sh PMNN H "$dimension" h4q
# #         DDP_NGPUS=1 ./train_JetClass.sh PMNN S "$dimension" h4q
        
#     done
# done
# 16D
# DDP_NGPUS=1 ./train_JetClass.sh PMNN R 16 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH 8 h4q

# 32D
# DDP_NGPUS=1 ./train_JetClass.sh PMNN R 32 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH 16 h4q

# DDP_NGPUS=1 ./train_JetClass.sh PMNN R 64 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH 32 h4q

# DDP_NGPUS=1 ./train_JetClass.sh PMNN R 128 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH 64 h4q




# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxS 8 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN HxS 8 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN S 16 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN H 16 h4q

# 32D
# DDP_NGPUS=1 ./train_JetClass.sh PMNN R 32 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH 16 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxS 16 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN HxS 16 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN S 32 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN H 32 h4q

# # 64D
# DDP_NGPUS=1 ./train_JetClass.sh PMNN R 64 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxH 32 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN RxS 32 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN HxS 32 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN S 64 h4q
# DDP_NGPUS=1 ./train_JetClass.sh PMNN H 64 h4q





# DDP_NGPUS=1 ./train_JetClass.sh PMNN R 8 R 64







# DDP_NGPUS=1 ./train_JetClass.sh PMNN S 8 R 64





