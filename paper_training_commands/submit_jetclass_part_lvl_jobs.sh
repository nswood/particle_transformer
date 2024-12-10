#!/bin/bash

# Define sets of dimensions for different configurations
# dimensions_set_1=(96 64 48 32)
# dimensions_set_2=(128 96 64 48 32)

dimensions_set_1=(96 64)
dimensions_set_2=(128 96 64)


dimensions_set_3=(128)
dimensions_set_4=(160)

dimensions_set_5=(144)
dimensions_set_6=(200)


# Training Parameters
curvatureInit=2
PMweightinit=1
interManFreq=2
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

#   First set of dimensions, full dimension command
  for dimension in "${dimensions_set_1[@]}"; do
    unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
    
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 20 --dev-id 'Euclidean_20_epochs' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans R ${dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"

    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans H $dimension R 16 kinpid --num-epochs 20 --dev-id 'tuned_paper_v3_hidden' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans H ${dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"

    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
  done

  # Second set of dimensions, half dimension commands
  for dimension in "${dimensions_set_2[@]}"; do
    half_dimension=$((dimension / 2))
    unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
#     command_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans RxH ${half_dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
    
#     command_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans RxS ${half_dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"

#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxh"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxs"
  done

  #Third set of dimensions, full dimension command with 30 epochs
  for dimension in "${dimensions_set_3[@]}"; do
    unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
  
    command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 30 --dev-id 'Euclidean_30_epochs_paper' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
  done

  # Fourth set of dimensions, half dimension commands with 30 epochs
  for dimension in "${dimensions_set_4[@]}"; do
    unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
  
    half_dimension=$((dimension / 2))
#     command_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans RxH ${half_dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
    
#     command_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans RxS ${half_dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"

#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxh"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxs"
  done
  
  for dimension in "${dimensions_set_5[@]}"; do
    unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
  
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 30 --dev-id 'Euclidean_30_epochs_paper' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans R ${dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 30 --dev-id 'Euclidean_30_epochs_paper' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans H ${dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
  done

  # Fourth set of dimensions, half dimension commands with 30 epochs
  for dimension in "${dimensions_set_6[@]}"; do
    half_dimension=$((dimension / 2))
    unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
    
#     command_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans RxH ${half_dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
    
    
#     command_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxh"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxs"
    command="DDP_NGPUS=2 COMMENT=${unique_id} ./train_JetClass.sh PMTrans RxS ${half_dimension} R 128 kinpid --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'part_lvl_redo' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers 4 --network-option num_heads 4 --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
  done
done







#--------------------------------------------------------------------------------



# # Repeat each job submission three times
# for repeat in {1..3}; do

#   First set of dimensions, full dimension command
#   for dimension in "${dimensions_set_1[@]}"; do
#     unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
    
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 20 --dev-id 'Euclidean_20_epochs' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
    
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans H $dimension R 16 kinpid --num-epochs 20 --dev-id 'tuned_paper_v3_hidden' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
#   done

#   # Second set of dimensions, half dimension commands
#   for dimension in "${dimensions_set_2[@]}"; do
#     half_dimension=$((dimension / 2))
#     unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
#     command_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
#     command_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxh"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxs"
#   done

#   #Third set of dimensions, full dimension command with 30 epochs
#   for dimension in "${dimensions_set_3[@]}"; do
#     unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
  
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 30 --dev-id 'Euclidean_30_epochs_paper' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
#   done

#   # Fourth set of dimensions, half dimension commands with 30 epochs
#   for dimension in "${dimensions_set_4[@]}"; do
#     unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
  
#     half_dimension=$((dimension / 2))
#     command_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
#     command_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxh"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxs"
#   done
  
#   for dimension in "${dimensions_set_5[@]}"; do
#     unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
  
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 30 --dev-id 'Euclidean_30_epochs_paper' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
    
#     command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 30 --dev-id 'Euclidean_30_epochs_paper' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
#   done

#   # Fourth set of dimensions, half dimension commands with 30 epochs
#   for dimension in "${dimensions_set_6[@]}"; do
#     half_dimension=$((dimension / 2))
#     unique_id="partD${dimension}_jetD128_layers4_nheads4_redo_${repeat}"
    
#     command_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
#     command_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxh"
#     sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxs"
#   done

# done
