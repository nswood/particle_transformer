#!/bin/bash
# submit_jobs.sh

DATA_DIR="/n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy_updated"
GEN_SCRIPT="manifold_gaussians/update_h5_with_graph_dist.py"  # Path to your update script

sub_dirs=("train" "test" "val")

# Loop over each specified subdirectory
for sub in "${sub_dirs[@]}"; do
    # Loop over all .h5 files in the current subdirectory
    for file in "${DATA_DIR}/${sub}"/*.h5; do
        # Create a unique job name based on the file name
        JOB_NAME="update_graph_dist_$(basename "${file}" .h5)"
        # Construct the command to run your update script with the current file
        CMD="python ${GEN_SCRIPT} --input_file ${file}"
        echo "Submitting job for ${file}"
        # Submit the job
        sbatch manifold_gaussians/submit_command.sh "${JOB_NAME}" "${CMD}"
    done
done

echo "All jobs submitted."
