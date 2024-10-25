#!/bin/bash

# Define sets of dimensions for different configurations
dimensions_set_1=(96 64 48 32)
dimensions_set_2=(128 96 64 48 32)
dimensions_set_3=(128)
dimensions_set_4=(160)

# Repeat each job submission three times
for repeat in {1..3}; do

  # First set of dimensions, full dimension command
  for dimension in "${dimensions_set_1[@]}"; do
    command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 20 --dev-id 'Euclidean_20_epochs' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
  done

  # Second set of dimensions, half dimension commands
  for dimension in "${dimensions_set_2[@]}"; do
    half_dimension=$((dimension / 2))
    command_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxh"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxs"
  done

  # Third set of dimensions, full dimension command with 30 epochs
  for dimension in "${dimensions_set_3[@]}"; do
    command="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans R $dimension R 16 kinpid --num-epochs 30 --dev-id 'Euclidean_30_epochs_paper' --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --att-metric 'tan_space' --base-resid-agg --base-activations 'act'"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_full_dim "$command"
  done

  # Fourth set of dimensions, half dimension commands with 30 epochs
  for dimension in "${dimensions_set_4[@]}"; do
    half_dimension=$((dimension / 2))
    command_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxH $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
    command_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_part_lvl ./train_JetClass.sh PMTrans RxS $half_dimension R 16 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'tuned_paper_v3_hidden_30_epochs' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxh"
    sbatch paper_training_commands/submit_generic_command.sh jc_p_half_dim "$command_rxs"
  done

done
