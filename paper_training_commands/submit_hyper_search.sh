#!/bin/bash

# Define single-value parameters directly
partD=240
jetD=256
layers=8
curvatureInit=2
PMweightinit=0.1
interManFreq=2
num_epochs=30
nheads=12
clamp=-1
# batchsize=32
batchsize=128
lrsched='flat+decay'
conv_embed="True"
betas="0.9,0.999"

#5e-4	0.2	radam	w.0	1	0.3FT	0.937254

# Parameters with multiple values
# lr=(5e-4 1e-3 1e-4)
# lr=(5e-3)
# dropout=(0.1 0.2)
# # optimizer=('riemann_ranger')
# optimizer=('r_adam' 'riemann_ranger')
# weightdecay=(1e-4 0)
# decaysteps=(0.7 0.5 0.3)
# gradaccum=(1 2 4)


lr=(5e-4)
dropout=(0.1 0.2)
optimizer=('r_adam')
weightdecay=(0)
decaysteps=(0.5)
# decaysteps=(0.5 0.7)
gradaccum=(1)
epochs=(30)
# curvature_init=(0.5 1 1.5 2 2.5)

#TopLandscape_kin_PMTrans_PMTrans_RxH_120_RxH_128_5e-4_0d2_r_adam_wd0_gradacc1_0d5_40_FT_performance_20241117_152106.csv




# # Repeat each job submission three times
for learning_rate in "${lr[@]}"; do
    for dropout_rate in "${dropout[@]}"; do
        for opt in "${optimizer[@]}"; do
            for w_decay in "${weightdecay[@]}"; do
                for decay_step in "${decaysteps[@]}"; do
                    for grad_accum in "${gradaccum[@]}"; do
                        for epoch in "${epochs[@]}"; do
    #                      for i in {1..1}; do
                    
#                         for c_init in "${curvature_init[@]}"; do
                            # Generate unique_id with all parameters
#                             unique_id="$(echo $c_init | sed 's/\./d/g')_FT"
                            unique_id="$(echo $learning_rate | sed 's/\./d/g')_$(echo $dropout_rate | sed 's/\./d/g')_${opt}_wd$(echo $w_decay | sed 's/\./d/g')_gradacc${grad_accum}_$(echo $decay_step | sed 's/\./d/g')_${epoch}_FT"
                            # hyper_scan_output_ft_v6
                            # Define the command without warmup steps
                            command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxH $((partD / 2)) RxH $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_long_trains' --att-metric 'tan_space' --num-epochs ${epoch} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${opt} --optimizer-option betas ${betas} --optimizer-option weight_decay ${w_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step}"
#                                 command_half_gvq="DDP_NGPUS=4 COMMENT=${unique_id} ./train_QuarkGluon.sh PMTrans RxH $((partD / 2)) RxH $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'testing_nans' --att-metric 'tan_space' --num-epochs ${epoch} --start-lr ${learning_rate} --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate ${dropout_rate} --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer ${opt} --optimizer-option betas ${betas} --optimizer-option weight_decay ${w_decay} --batch-size ${batchsize} --grad-accum ${grad_accum} --decay-steps ${decay_step}"

                            # Submit the job
                            sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"
#                                 sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_gvq"
#                             done
                        done
                    done
                done
            done
        done
    done
done




command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxH $((partD / 2)) RxH $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_long_trains' --att-metric 'tan_space' --num-epochs 40 --start-lr 5e-4 --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate 0.2 --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer r_adam --optimizer-option betas ${betas} --optimizer-option weight_decay 0 --batch-size 64 --grad-accum 1 --decay-steps 0.5 --model-prefix training/TopLandscape/PMTrans/20241117-152056_example_PMTransformer_f007342b14578bfa390342230e5d90a4_r_adam_lr0.0005_batch128PMTrans_RxH_120_RxH_128_5e-4_0d2_r_adam_wd0_gradacc1_0d5_40_FT/net --load-epoch 29 --fix-c-after-load"
# sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"

# command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxH $((partD / 2)) RxH $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_long_trains' --att-metric 'tan_space' --num-epochs 50 --start-lr 5e-4 --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate 0.2 --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer r_adam --optimizer-option betas ${betas} --optimizer-option weight_decay 0 --batch-size 128 --grad-accum 1 --decay-steps 0.5 --model-prefix training/TopLandscape/PMTrans/20241117-152056_example_PMTransformer_f007342b14578bfa390342230e5d90a4_r_adam_lr0.0005_batch128PMTrans_RxH_120_RxH_128_5e-4_0d2_r_adam_wd0_gradacc1_0d5_40_FT/net --load-epoch 36 --fix-c-after-load"
# sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"


######################################

# #TopLandscape_kin_PMTrans_PMTrans_RxH_120_RxH_128_5e-4_0d1_r_adam_wd0_gradacc1_0d5_40_FT_performance_20241117_152108.csv
command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxH $((partD / 2)) RxH $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_long_trains' --att-metric 'tan_space' --num-epochs 50 --start-lr 5e-4 --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate 0.1 --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer r_adam --optimizer-option betas ${betas} --optimizer-option weight_decay 0 --batch-size 128 --grad-accum 1 --decay-steps 0.5 --model-prefix training/TopLandscape/PMTrans/20241117-152057_example_PMTransformer_753b9e9353988c4a08b6ae2a4a7ec58b_r_adam_lr0.0005_batch128PMTrans_RxH_120_RxH_128_5e-4_0d1_r_adam_wd0_gradacc1_0d5_40_FT/net --load-epoch 35 --fix-c-after-load"
# sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"


# #TopLandscape_kin_PMTrans_PMTrans_RxH_120_RxH_128_5e-4_0d2_r_adam_wd0_gradacc1_0d7_40_FT_performance_20241117_152107.csv
command_half_top="DDP_NGPUS=2 COMMENT=${unique_id} ./train_TopLandscape.sh PMTrans RxH $((partD / 2)) RxH $((jetD / 2)) kin --PM-weight-initialization-factor ${PMweightinit} --inter-man-att ${interManFreq} --inter-man-att-method 'v3' --dev-id 'top_long_trains' --att-metric 'tan_space' --num-epochs 50 --start-lr 5e-4 --base-resid-agg --base-activations 'act' --network-option num_layers ${layers} --network-option num_heads ${nheads} --network-option dropout_rate 0.2 --network-option curvature_init ${curvatureInit} --network-option conv_embed ${conv_embed} --network-option clamp ${clamp} --optimizer r_adam --optimizer-option betas ${betas} --optimizer-option weight_decay 0 --batch-size 128 --grad-accum 1 --decay-steps 0.7 --model-prefix training/TopLandscape/PMTrans/20241117-152056_example_PMTransformer_f007342b14578bfa390342230e5d90a4_r_adam_lr0.0005_batch128PMTrans_RxH_120_RxH_128_5e-4_0d2_r_adam_wd0_gradacc1_0d7_40_FT/net --load-epoch 37 --fix-c-after-load"
# sbatch paper_training_commands/submit_scan_jobs.sh jc_j_rxh "$command_half_top"






#     #     command_half_qvg="DDP_NGPUS=4 COMMENT=qvg_benchmarking_8_heads ./train_QuarkGluon.sh PMTrans RxH $half_dimension RxH 48 kinpid --PM-weight-initialization-factor 1 --inter-man-att 0 --inter-man-att-method 'v3' --dev-id 'qvg_benchmarking_8_heads' --att-metric 'tan_space' --num-epochs 20 --base-resid-agg --base-activations 'act' --network-option num_layers 8 --network-option num_heads 8 --network-option dropout_rate 0.2"
#     #     sbatch paper_training_commands/submit_generic_command.sh jc_j_rxh "$command_half_qvg"


