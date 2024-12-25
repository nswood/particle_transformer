#!/bin/bash
# dimensionsSingle=(4 8 16)  # Add your dimensions here
dimensionsSingle=(2)  # Add your dimensions here

epochs=30
lr=1e-4
opt='r_adam'
decay=0.5

for i in {1..1}; do

#     Loop over each dimension in the list
    for dimension in "${dimensionsSingle[@]}"; do

        unique_id="test_H_${dimension}_${i}"
        command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_TopLandscape.sh PMNN H  $dimension kin --dev-id 'PMMLP_top_testing' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --batch-size 128"
        sbatch paper_training_commands/submit_MLP_jobs.sh jc_j_rxh "$command_h"
    done
    
done



