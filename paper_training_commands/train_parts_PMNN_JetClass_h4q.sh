#!/bin/bash
dimensionsSingle=(2 4 8 16 32)  # Add your dimensions here

# dimensionsSingle=(4)  # Add your dimensions here
dimensions=(4 8 16 32)  # Add your dimensions here
# dimensions=(4)  # Add your dimensions here

decay=0.7
# epochs=50
epochs=50
lr=1e-3
opt='r_adam'


# unique_id="R_2_output"
# command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R 2 h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test --predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241122-225952_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_2_h4q_R_2_2_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"

# unique_id="H_2_output"
# command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H 2 h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test --predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241123-000404_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_2_h4q_H_2_5_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"

# unique_id="H_4_output"
# command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H 4 h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test --predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241122-225952_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_4_h4q_H_4_2_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"

# unique_id="R_4_output"
# command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R 4 h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test --predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241122-231227_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_4_h4q_R_4_3_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"

# unique_id="H_8_output"
# command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H 8 h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test --predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241120-005828_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_8_h4q_H_8_10/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"

# unique_id="R_8_output"
# command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R 8 h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test --predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241119-200500_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_8_h4q_R_8_4/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"


# unique_id="H_8_output"
# command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H 8 h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test --predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241123-014958_example_PMNN_r_adam_lr0.001_batch1024PMNN_H_8_h4q_H_8_10_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"


# unique_id="R_8_output"
# command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R 8 h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test --predict --predict-part-embed --model-prefix /n/holystore01/LABS/iaifi_lab/Lab/nswood/training/JetClass/Pythia/h4q/PMNN/20241122-235137_example_PMNN_r_adam_lr0.001_batch1024PMNN_R_8_h4q_R_8_4_new_data/net"
# sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"





# # for i in {1..3}; do
for i in {1..10}; do

#     Loop over each dimension in the list
    for dimension in "${dimensionsSingle[@]}"; do
        half_dimension=$((dimension / 2))  # Calculate half of the dimension

        # Full-dimension commands
        unique_id="S_${dimension}_${i}_new_data"
        command_s="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN S $dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
        sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_s"

        unique_id="R_${dimension}_${i}_new_data"
        command_r="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN R $dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_r"

        unique_id="H_${dimension}_${i}_new_data"
        command_h="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN H $dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_h"
    done
    
    for dimension in "${dimensions[@]}"; do
        half_dimension=$((dimension / 2))  # Calculate half of the dimension

        # Define commands with half dimensions and unique identifiers
        unique_id="RxH_${dimension}_${i}_new_data"
        command_rxh="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN RxH $half_dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
        
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_rxh"

        unique_id="HxH_${dimension}_${i}_new_data"
        command_hxh="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN HxH $half_dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_hxh"

        unique_id="SxS_${dimension}_${i}_new_data"
        command_sxs="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN SxS $half_dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
        sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_sxs"

        unique_id="RxR_${dimension}_${i}_new_data"
        command_rxr="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN RxR $half_dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
#         sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_rxr"

        unique_id="RxS_${dimension}_${i}_new_data"
        command_rxs="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN RxS $half_dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
        sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_rxs"

        unique_id="HxS_${dimension}_${i}_new_data"
        command_hxs="DDP_NGPUS=1 COMMENT=${unique_id} ./train_JetClass.sh PMNN HxS $half_dimension h4q --dev-id 'PMNN_h4q_neurips_for_paper_new_data_sphere' --start-lr ${lr} --num-epochs ${epochs} --PM-weight-initialization-factor 0.1 --decay-steps ${decay} --optimizer ${opt} --skip-test "
        sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_hxs"

        
    done
done



