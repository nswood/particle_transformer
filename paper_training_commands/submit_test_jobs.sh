#!/bin/bash

# Define dimensions for this job
dimensions=(32)

# Repeat each job submission three times

# Loop over each dimension in the list
half_dimension=$((dimension / 2))  # Calculate half of the dimension


# Half dimension commands
    command_half_rxh="DDP_NGPUS=4 COMMENT=final_jetclass_jet_lvl ./train_JetClass.sh PMTrans R 32 RxH $half_dimension kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'jet_lvl_v3_hidden' --att-metric 'tan_space' --num-epochs 30 --base-resid-agg --base-activations 'act'"

echo $command_half_rxh

sbatch paper_training_commands/test_submit_generic_command.sh jc_j_rxh "$command_half_rxh"


