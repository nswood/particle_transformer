#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=1
samples_per_epoch=$((100 * 256 / $NGPUS))
samples_per_epoch_val=$((20 * 256))
dataopts="--num-workers 1 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
PART_GEOM=$2
PART_DIM=$3
JET_GEOM=$4
JET_DIM=$5

if [[ "$model" == "PMTrans" ]]; then
    modelopts="networks/example_PMTransformer_embedder.py --use-amp --optimizer-option weight_decay 0.01 --part-geom ${PART_GEOM} --part-dim ${PART_DIM} --jet-geom ${JET_GEOM} --jet-dim ${JET_DIM}"
    suffix=${model}_${PART_GEOM}_${PART_DIM}_${JET_GEOM}_${JET_DIM}
    batchopts="--batch-size 256 --start-lr 1e-4"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=$6
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"
[[ -z ${PART_GEOM} ]] && PART_GEOM="R"
[[ -z ${PART_DIM} ]] && PART_DIM=64
[[ -z ${JET_GEOM} ]] && JET_GEOM="R"
[[ -z ${JET_DIM} ]] && JET_DIM=8

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# currently only Pythia
SAMPLE_TYPE=Pythia

$CMD \
    --data-train \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/train_100M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/train_100M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/train_100M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/train_100M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/train_100M/ZJetsToNuNu_*.root" \
    --data-val "${DATADIR}/${SAMPLE_TYPE}/val_5M/*.root" \
    --data-test \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToBB_100.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToCC_100.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToGG_100.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToWW2Q1L_100.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToWW4Q_100.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/test_20M/TTBar_100.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/test_20M/TTBarLep_100.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/test_20M/WToQQ_100.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/test_20M/ZToQQ_100.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/test_20M/ZJetsToNuNu_100.root" \
    --data-config data/JetClass/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer rlion --log-dir JetClass_logs/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${suffix} --embedding-mode \
    "${@:7}"
