#!/bin/bash

# Define single-value parameters directly
curvatureInit=2
PMweightinit=1
interManFreq=2
# interManFreq=-1
num_epochs=30
clamp=-1
batchsize=128
lrsched='flat+decay'
conv_embed="True"
# conv_embed="False"
betas="0.9,0.999"
learning_rate=5e-4
dropout_rate=0.1
optimizer='r_adam'
weight_decay=0
decay_step=0.5
grad_accum=1
epochs=30


# List of sets for partD, layers, and nheads
configurations=(
    "10 24 2 1"
    "4 12 2 1"
    "4 4 1 1"
    "2 2 1 1"
)

# Loop over configurations
for i in {1..2}; do
    for config in "${configurations[@]}"; do
    # Extract individual values from the configuration
        read -r partD jetD layers nheads<<< "$config"
        # Generate unique_id with all parameters
        unique_id="partD${partD}_jetD${jetD}_layers${layers}_nheads${nheads}_final_batch_redo_${i}"

        # Single geometries
        # (R,R)

        command_half_top="DDP_NGPUS=1 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans R ${partD} R ${jetD} kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_paper_small' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"
        
        sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"

        # (H,H)
        command_half_top="DDP_NGPUS=1 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans H ${partD} H ${jetD} kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_paper_small' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"

        sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"

    done
done



# Loop over configurations

configurations=(
    "10 24 2 1"
    "4 12 2 1"
    "4 4 1 1"
)


for i in {1..2}; do
    for config in "${configurations[@]}"; do
    # Extract individual values from the configuration
        read -r partD jetD layers nheads<<< "$config"
        # Generate unique_id with all parameters
        unique_id="partD${partD}_jetD${jetD}_layers${layers}_nheads${nheads}_final_batch_redo_${i}"

        # # # # # Submit the jobs
        # sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"

        # # (RxR, RxR)
        command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxR $((partD / 2)) RxR $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_paper_small' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"

        sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"

        # (RxH, RxH)
        command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxH $((partD / 2)) RxH $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_paper_small' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step} --network-option pair_embed_dims None"

        sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"
    done
done


configurations=(
    "128 96 8 4"
    "80 96 8 4"
    "40 64 8 2"
    "20 32 8 2"

)

# Loop over configurations
for i in {1..2}; do
    for config in "${configurations[@]}"; do
    # Extract individual values from the configuration
        read -r partD jetD layers nheads<<< "$config"
        # Generate unique_id with all parameters
        unique_id="partD${partD}_jetD${jetD}_layers${layers}_nheads${nheads}_final_batch_redo_${i}"

        # -----------------------------------------------------
        # Large Models
        # -----------------------------------------------------

        # (H,H)
        command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans H ${partD} H ${jetD} kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_paper_large' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step}"

        sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"

         # (R,R)
        command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans R ${partD} R ${jetD} kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_paper_large' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step}"

        sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"
    done
done

configurations=(
    "160 128 8 4"
    "128 96 8 4"
    "80 96 8 4"
    "40 64 8 2"
)

# Loop over configurations
for i in {1..2}; do
    for config in "${configurations[@]}"; do
    # Extract individual values from the configuration
        read -r partD jetD layers nheads<<< "$config"
        # Generate unique_id with all parameters
        unique_id="partD${partD}_jetD${jetD}_layers${layers}_nheads${nheads}_final_batch_redo_${i}"

        # -----------------------------------------------------
        # Large Models
        # -----------------------------------------------------

        # (RxR,RxR)
        command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxR $((partD / 2)) RxR $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_paper_large' --att-metric 'tan_space' --num-epochs ${epochs} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${optimizer} --optimizer-option betas ${betas} --optimizer-option weight_decay ${weight_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step}"

        sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"

    done
done

