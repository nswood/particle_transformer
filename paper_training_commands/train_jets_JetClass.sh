#!/bin/bash
#SBATCH --job-name=JC_jets
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
export WORLD_SIZE=2

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

# 4D
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 R 4 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxH 2 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxS 2 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 HxS 2 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 S 4 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 H 4 full

# 8D
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 R 8 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxH 4 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxS 4 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 HxS 4 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 S 8 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 H 8 full

# 16D
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 R 16 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxH 8 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxS 8 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 HxS 8 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 S 8 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 H 8 full

# 32D
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 R 32 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxH 16 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxS 16 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 HxS 16 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 S 32 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 H 32 full

# 64D
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 R 64 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxH 32 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 RxS 32 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 HxS 32 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 S 64 full
DDP_NGPUS=2 ./train_JetClass.sh PMTrans R 32 H 64 full

# DDP_NGPUS=1 ./train_JetClass.sh PMTrans R 8 R 16 full full
# DDP_NGPUS=1 ./train_JetClass.sh PMTrans RxH 4 R 16 full full
# DDP_NGPUS=1 ./train_JetClass.sh PMTrans RxS 4 R 16 full full
# DDP_NGPUS=1 ./train_JetClass.sh PMTrans HxS 4 R 16 full full
# DDP_NGPUS=1 ./train_JetClass.sh PMTrans S 8 R 16 full full
# DDP_NGPUS=1 ./train_JetClass.sh PMTrans H 8 R 16 full full









