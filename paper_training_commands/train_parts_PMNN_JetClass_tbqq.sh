#!/bin/bash
dimensionsSingle=(2 4 8 16 32)  # Add your dimensions here

# dimensionsSingle=(4)  # Add your dimensions here
dimensions=(4 8 16 32)  # Add your dimensions here
# dimensions=(4)  # Add your dimensions here

decay=0.7
epochs=50
lr=1e-3
opt='r_adam'


# unique_id="R_2_output"
# command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R 2 tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test--predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/tbqq/PMNN/20241120-000628_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_2_tbqq_R_2_9/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"

# unique_id="H_2_output"
# command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H 2 tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test--predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/tbqq/PMNN/20241119-200325_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_2_tbqq_H_2_3/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"

# unique_id="H_4_output"
# command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H 4 tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test--predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/tbqq/PMNN/20241123-043033_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_4_tbqq_H_4_8_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"

# unique_id="R_4_output"
# command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R 4 tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test--predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/tbqq/PMNN/20241123-032930_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_4_tbqq_R_4_5_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"

# unique_id="H_8_output"
# command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H 8 tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test--predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/tbqq/PMNN/20241120-005828_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_8_tbqq_H_8_10/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"

# unique_id="R_8_output"
# command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R 8 tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test--predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/tbqq/PMNN/20241119-200500_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_8_tbqq_R_8_4/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"


# unique_id="H_8_output"
# command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H 8 tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test--predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/tbqq/PMNN/20241123-045217_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_8_tbqq_H_8_9_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"

# unique_id="R_8_output"
# command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R 8 tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test--predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/tbqq/PMNN/20241123-020647_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_8_tbqq_R_8_1_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"








# # for i in {11..15}; do
for i in {1..10}; do

    # Loop over each dimension in the list
    for dimension in "${dimensionsSingle[@]}"; do
        half_dimension=$((dimension / 2))  # Calculate half of the dimension

        # Full-dimension commands
        unique_id="S_${dimension}_${i}_new_data"
        command_s="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN S $dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt}"
        sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_s"

        unique_id="R_${dimension}_${i}_new_data"
        command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R $dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test"
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"

        unique_id="H_${dimension}_${i}_new_data"
        command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H $dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test"
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"
    done
    
    for dimension in "${dimensions[@]}"; do
        half_dimension=$((dimension / 2))  # Calculate half of the dimension

        # Define commands with half dimensions and unique identifiers
        unique_id="RxH_${dimension}_${i}_new_data"
        command_rxh="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN RxH $half_dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test"
        
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_rxh"

        unique_id="HxH_${dimension}_${i}_new_data"
        command_hxh="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN HxH $half_dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test"
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_hxh"

        unique_id="SxS_${dimension}_${i}_new_data"
        command_sxs="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN SxS $half_dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test"
        sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_sxs"

        unique_id="RxR_${dimension}_${i}_new_data"
        command_rxr="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN RxR $half_dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test"
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_rxr"

        unique_id="RxS_${dimension}_${i}_new_data"
        command_rxs="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN RxS $half_dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test"
        sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_rxs"

        unique_id="HxS_${dimension}_${i}_new_data"
        command_hxs="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN HxS $half_dimension tbqq --dev-id 'PMNN_tbqq_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test"
        sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_hxs"

        
    done
done



