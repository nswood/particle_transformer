#!/bin/bash
#SBATCH --job-name=JC_p
#SBATCH --partition=gpu
#SBATCH --time=72:00:00

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

# # 8D
# DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans R 8 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans RxH 4 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans RxS 4 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans HxS 4 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans S 8 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans H 8 R 16 kinpid

dimensions=(8 16 32 64)  # Add your dimensions here

# Loop over each dimension in the list
for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Command with full dimension
    DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans R "$dimension" R 16 kinpid
    DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans H "$dimension" R 16 kinpid
    DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans S "$dimension" R 16 kinpid
    
    # Commands with dimensions / 2
    DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans RxH "$half_dimension" R 16 kinpid
    DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans RxS "$half_dimension" R 16 kinpid
    DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans HxS "$half_dimension" R 16 kinpid
    DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans RxR "$half_dimension" R 16 kinpid
    
done

# # 16D
# DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans S 16 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh TestPMTrans H 16 R 16 kinpid

# # 32D
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans R 32 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans RxH 16 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans RxS 16 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans HxS 16 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans S 32 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans H 32 R 16 kinpid

# # # 64D
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans R 64 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans RxH 32 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans RxS 32 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans HxS 32 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans S 64 R 16 kinpid
# DDP_NGPUS=4 ./train_JetClass.sh PMTrans H 64 R 16 kinpid





# DDP_NGPUS=4 ./train_JetClass.sh PMTrans R 8 R 64







# DDP_NGPUS=4 ./train_JetClass.sh PMTrans S 8 R 64





