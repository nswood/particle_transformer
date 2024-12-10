#!/bin/bash

# Define dimensions for this job
dimensions=(16 32 48 64)

# Training Parameters
curvatureInit=2
PMweightinit=1
interManFreq=2
num_epochs=30
clamp=-1
batchsize=128
lrsched='flat+decay'
conv_embed="True"
betas="0.9,0.999"
learning_rate=5e-4
dropout_rate=0.1
optimizer='r_adam'
weight_decay=0
decay_step=0.5
grad_accum=1
epochs=20


# Repeat each job submission three times
for repeat in {1..3}; do

  # Loop over each dimension in the list
  for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension
    unique_id="partD32_jetD${dimension}_layers4_nheads4_redo_${repeat}"

    # Full dimension commands
#     command_full_r="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 R $dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans R 32 R ${dimension} kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'jet_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_r "$command"

    
#     command_full_h="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 H $dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans R 32 H ${dimension} kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'jet_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_r "$command"
    
#     command_full_s="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 S $dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans R 32 S ${dimension} kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'jet_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_r "$command"

#     sbatch paper_training_commands/submit_generic_command.sh jc_j_r "$command_full_r"
#     sbatch paper_training_commands/submit_generic_command.sh jc_j_h "$command_full_h"
#     sbatch paper_training_commands/submit_generic_command.sh jc_j_s "$command_full_s"

    # Half dimension commands
#     command_half_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 RxH $half_dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans R 32 RxH ${half_dimension} kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'jet_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_r "$command"
    
#     command_half_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 RxS $half_dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans R 32 RxS ${half_dimension} kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'jet_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_r "$command"
    
#     command_half_hxs="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 HxS $half_dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans R 32 HxS ${half_dimension} kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'jet_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_r "$command"

#     sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_half_rxh"
#     sbatch paper_training_commands/submit_generic_command.sh jc_j_rxs "$command_half_rxs"
#     sbatch paper_training_commands/submit_generic_command.sh jc_j_hxs "$command_half_hxs"

  done

done
