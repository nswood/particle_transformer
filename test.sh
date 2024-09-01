#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu_test
#SBATCH --time=8:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
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

# DDP_NGPUS=1 ./train_JetClass.sh PMTrans R 32 R 4 full
# DDP_NGPUS=1 ./train_JetClass.sh PMTrans R 32 H 4 full

# DDP_NGPUS=1 ./train_JetClass.sh PMTrans R 32 H 8 full

# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 16 R 4
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTransMod R 32 H 4




# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 R 8
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 S 8
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 RxH 4
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 RxS 4
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 HxS 4


# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 R 16
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 H 16
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 S 16
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 RxH 8
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 RxS 8
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 HxS 8


# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 R 32
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 H 32
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 S 32
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 RxH 16
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 RxS 16
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 HxS 16

# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 R 64
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 H 64
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 S 64
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 RxH 32
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 RxS 32
# DDP_NGPUS=1 ./train_TopLandscape.sh PMTrans R 64 HxS 32


