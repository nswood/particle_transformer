#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_TopLandscape`
DATADIR=${DATADIR_TopLandscape}
[[ -z $DATADIR ]] && DATADIR='./datasets/TopLandscape'
# set a comment via `COMMENT`
suffix=${COMMENT}


# PN, PFN, PCNN, ParT
model=$1
PART_GEOM=$2
PART_DIM=$3
JET_GEOM=$4
JET_DIM=$5


# "kin"
FEATURE_TYPE=$6
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kin"
if [[ "${FEATURE_TYPE}" != "kin" ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi


# Default values
[[ -z ${PART_GEOM} ]] && PART_GEOM="R"
[[ -z ${PART_DIM} ]] && PART_DIM=64
[[ -z ${JET_GEOM} ]] && JET_GEOM="R"
[[ -z ${JET_DIM} ]] && JET_DIM=64


extraopts=""

if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "PMTrans" ]]; then
    modelopts="networks/example_PMTransformer.py --use-amp --optimizer-option weight_decay 0.01 --part-geom ${PART_GEOM} --part-dim ${PART_DIM} --jet-geom ${JET_GEOM} --jet-dim ${JET_DIM}"
    suffix=${model}_${PART_GEOM}_${PART_DIM}_${JET_GEOM}_${JET_DIM}
    lr="1e-3"
elif [[ "$model" == "PMTransMod" ]]; then
    modelopts="networks/example_PMTransformer_modified.py --use-amp --optimizer-option weight_decay 0.01 --part-geom ${PART_GEOM} --part-dim ${PART_DIM} --jet-geom ${JET_GEOM} --jet-dim ${JET_DIM}"
    suffix=${model}_${PART_GEOM}_${PART_DIM}_${JET_GEOM}_${JET_DIM}
    lr="1e-5"
elif [[ "$model" == "PMTransBench" ]]; then
    modelopts="networks/example_PMTransformer_Benchmarks.py --use-amp --optimizer-option weight_decay 0.01 --part-geom ${PART_GEOM} --part-dim ${PART_DIM} --jet-geom ${JET_GEOM} --jet-dim ${JET_DIM}"
    suffix=${model}_${PART_GEOM}_${PART_DIM}_${JET_GEOM}_${JET_DIM}
    lr="1e-3"
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="networks/example_ParticleTransformer_finetune.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-4"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none --load-model-weights models/ParT_kin.pt"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    lr="1e-2"
elif [[ "$model" == "PN-FineTune" ]]; then
    modelopts="networks/example_ParticleNet_finetune.py"
    lr="1e-3"
    extraopts="--optimizer-option lr_mult (\"fc_out.*\",50) --lr-scheduler none --load-model-weights models/ParticleNet_kin.pt"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
else
    echo "Invalid model $model!"
    exit 1
fi

weaver \
    --data-train "${DATADIR}/train_file.parquet" \
    --data-val "${DATADIR}/val_file.parquet" \
    --data-test "${DATADIR}/test_file.parquet" \
    --data-config data/TopLandscape/top_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/TopLandscape/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 1 --in-memory \
    --batch-size 256 --samples-per-epoch $((4800 * 256)) --samples-per-epoch-val $((1600 * 256)) --num-epochs 10 --gpus 0 \
    --start-lr $lr --optimizer rlion --log Top_logs/TopLandscape_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard TopLandscape_${FEATURE_TYPE}_${model}_${suffix} \
    ${extraopts} "${@:6}"
