#!/bin/bash

# Define dimensions for this job
dimensions=(16 32 48 64)

# Repeat each job submission three times
for repeat in {1..3}; do

  # Loop over each dimension in the list
  for dimension in "${dimensions[@]}"; do
    half_dimension=$((dimension / 2))  # Calculate half of the dimension

    # Full dimension commands
    command_full_r="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 R $dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command_full_h="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 H $dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command_full_s="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 S $dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"

    sbatch paper_training_commands/submit_generic_command.sh jc_j_r "$command_full_r"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_h "$command_full_h"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_s "$command_full_s"

    # Half dimension commands
    command_half_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 RxH $half_dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command_half_rxs="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 RxS $half_dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"
    command_half_hxs="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 HxS $half_dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act'"

    sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_half_rxh"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_rxs "$command_half_rxs"
    sbatch paper_training_commands/submit_generic_command.sh jc_j_hxs "$command_half_hxs"

  done

done
