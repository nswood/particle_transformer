#!/bin/bash

# # Define dimensions for this job
# partD=(192)
# jetD=(48)

# # Repeat each job submission three times
# for repeat in {1..2 }; do

#   # Loop over each dimension in the list
#   for dimension in "${partD[@]}"; do
#     half_dimension=$((dimension / 2))  # Calculate half of the dimension

#     # Half dimension commands
#     command_half_top="DDP_NGPUS=4 COMMENT=top_benchmarking_8_heads ./train_TopLandscape.sh PMTrans RxH $half_dimension RxH 48 kin --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'top_benchmarking_8_heads' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act' --network-option num_layers 8 --network-option num_heads 8 --network-option dropout_rate 0.2"
#     command_half_qvg="DDP_NGPUS=4 COMMENT=qvg_benchmarking_8_heads ./train_QuarkGluon.sh PMTrans RxH $half_dimension RxH 48 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'qvg_benchmarking_8_heads' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act' --network-option num_layers 8 --network-option num_heads 8 --network-option dropout_rate 0.2"
    
#     sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_half_top"
#     sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_half_qvg"
    

#   done

# done

part_dim=240
jet_dim=96
num_heads=8
num_layers=8
learning_rate=0.001
dropout_rate=0.1
curvature_init=2
pm_weight_init=1

half_jetD=$((jet_dim / 2))  # Calculate half of the jet dimension
half_partD=$((part_dim / 2))  # Calculate half of the part dimension


# Create a unique identifier with '.' replaced by 'd'
unique_id="${part_dim}_${jet_dim}_${num_heads}_${num_layers}_$(echo $learning_rate | sed 's/\./d/g')_$(echo $dropout_rate | sed 's/\./d/g')_$(echo $curvature_init | sed 's/\./d/g')_$(echo $pm_weight_init | sed 's/\./d/g')_run${repeat}"

command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxH $half_partD RxH $half_jetD kin --PM-weight-initialization-factor ${pm_weight_init} --inter-man-att 0 --inter-man-att-method 'v3' --dev-id '${unique_id}' --att-metric 'tan_space' --num-epochs 5 --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${num_layers} --network-option num_heads ${num_heads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvature_init}"

# Submit the job
sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"
