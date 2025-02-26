#!/bin/bash
# submit_jobs.sh

# Hardcoded parameters
# OUTDIR_BASE="/n/holystore01/LABS/iaifi_lab/Lab/nswood/testing_hyperbolic_gaussians_toy"
OUTDIR_BASE="/n/holystore01/LABS/iaifi_lab/Lab/nswood/hyperbolic_gaussians_toy"
GEN_SCRIPT="manifold_gaussians/generate_toy_data.py"  # Path to your data generation script

# Generation parameters
N_CENTROIDS=4
CURVATURE=-1

# Batch parameters: each job produces 25k total samples.
BATCH_TOTAL=1000
# Points per centroid (n_samples per centroid) is:
N_SAMPLES_PER_CENTROID=100  # 6250

# Total sizes for each split
TRAIN_TOTAL=200000
TEST_TOTAL=25000
VAL_TOTAL=25000

# Batch parameters: each job produces 25k total samples.
# BATCH_TOTAL=50
# # Points per centroid (n_samples per centroid) is:
# N_SAMPLES_PER_CENTROID=100  # 6250

# # Total sizes for each split
# TRAIN_TOTAL=100
# TEST_TOTAL=50
# VAL_TOTAL=50

# Calculate number of batches/jobs per split
TRAIN_BATCHES=$(( TRAIN_TOTAL / BATCH_TOTAL ))
TEST_BATCHES=$(( TEST_TOTAL / BATCH_TOTAL ))
VAL_BATCHES=$(( VAL_TOTAL / BATCH_TOTAL ))

echo "Submitting jobs with the following configuration:"
echo "  - Train: $TRAIN_BATCHES jobs (200k samples)"
echo "  - Test:  $TEST_BATCHES job  (25k samples)"
echo "  - Val:   $VAL_BATCHES job  (25k samples)"
echo "Each job: ${N_CENTROIDS} centroids with ${N_SAMPLES_PER_CENTROID} points per centroid (total ${BATCH_TOTAL} points)"
echo "Curvature: ${CURVATURE}, STD: ${STD}"
echo "Output base directory: ${OUTDIR_BASE}"

# Submit train jobs
# for ((i=1; i<=TRAIN_BATCHES; i++)); do
#     JOB_NAME="train_${i}"
#     # Save train data under a dedicated subfolder
#     OUTDIR="${OUTDIR_BASE}/train"
#     CMD="python ${GEN_SCRIPT} --n_datapoints ${BATCH_TOTAL} --n_samples ${N_SAMPLES_PER_CENTROID} --n_centroids ${N_CENTROIDS} --curvature ${CURVATURE} --outdir ${OUTDIR} --file_name ${JOB_NAME}"
#     echo "Submitting job: ${JOB_NAME}"
#     sbatch manifold_gaussians/submit_command.sh ${JOB_NAME} "$CMD"
# done

# Submit test jobs
for ((i=1; i<=TEST_BATCHES; i++)); do
    JOB_NAME="test_${i}"
    OUTDIR="${OUTDIR_BASE}/test"
    CMD="python ${GEN_SCRIPT} --n_datapoints ${BATCH_TOTAL} --n_samples ${N_SAMPLES_PER_CENTROID} --n_centroids ${N_CENTROIDS} --curvature ${CURVATURE} --outdir ${OUTDIR} --file_name ${JOB_NAME}"
    echo "Submitting job: ${JOB_NAME}"
    sbatch manifold_gaussians/submit_command.sh ${JOB_NAME} "$CMD"
done

# Submit validation jobs
for ((i=1; i<=VAL_BATCHES; i++)); do
    JOB_NAME="val_${i}"
    OUTDIR="${OUTDIR_BASE}/val"
    CMD="python ${GEN_SCRIPT} --n_datapoints ${BATCH_TOTAL} --n_samples ${N_SAMPLES_PER_CENTROID} --n_centroids ${N_CENTROIDS} --curvature ${CURVATURE} --outdir ${OUTDIR} --file_name ${JOB_NAME}"
    echo "Submitting job: ${JOB_NAME}"
    sbatch manifold_gaussians/submit_command.sh ${JOB_NAME} "$CMD"
done

echo "All jobs submitted."
