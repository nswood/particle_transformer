#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_QuarkGluon`
DATADIR=${DATADIR_QuarkGluon}
[[ -z $DATADIR ]] && DATADIR='./datasets/QuarkGluon'

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
# "kin", "kinpid", "kinpidplus"
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kinpid"

if [[ "${FEATURE_TYPE}" == "kin" ]]; then
    pretrain_type="kin"
elif [[ "${FEATURE_TYPE}" =~ ^(kinpid|kinpidplus)$ ]]; then
    pretrain_type="kinpid"
else
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

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
    lr="1e-3"
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="networks/example_ParticleTransformer_finetune.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-4"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    lr="1e-2"
elif [[ "$model" == "PN-FineTune" ]]; then
    modelopts="networks/example_ParticleNet_finetune.py"
    lr="1e-3"
    extraopts="--optimizer-option lr_mult (\"fc_out.*\",50) --lr-scheduler none"
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



if [[ "$model" == "ParT-FineTune" ]]; then
    modelopts+=" --load-model-weights models/ParT_${pretrain_type}.pt"
fi
if [[ "$model" == "PN-FineTune" ]]; then
    modelopts+=" --load-model-weights models/ParticleNet_${pretrain_type}.pt"
fi

weaver \
    --data-train "${DATADIR}/train_file_*.parquet" \
    --data-test "${DATADIR}/test_file_*.parquet" \
    --data-config data/QuarkGluon/qg_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/QuarkGluon/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
    --batch-size 256 --samples-per-epoch 1600000 --samples-per-epoch-val 200000 --num-epochs 10 --gpus 0 \
    --start-lr $lr --optimizer rlion --log logs/QuarkGluon_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard QuarkGluon_${FEATURE_TYPE}_${model}${suffix} \
    ${extraopts} "${@:7}"
