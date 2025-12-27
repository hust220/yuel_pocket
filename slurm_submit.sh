#!/bin/bash

# Usage: bash slurm_submit.sh scripts/train.sh JOB_NAME
# Example: bash slurm_submit.sh scripts/train_dist_c4.sh my_training_job

if [ $# -ne 2 ]; then
    echo "Usage: $0 <script_path> <job_name>"
    echo "Example: $0 scripts/train_dist_c4.sh my_training_job"
    exit 1
fi

SCRIPT_PATH=$1
JOB_NAME=$2

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Submit the job
sbatch --partition=gpu \
       --gres=gpu:1 \
       --mem=64G \
       --cpus-per-task=16 \
       --time=48:00:00 \
       --account=dokhlab \
       --job-name="$JOB_NAME" \
       --output="slurm_logs/${JOB_NAME}_%j.out" \
       --error="slurm_logs/${JOB_NAME}_%j.err" \
       --wrap="bash $SCRIPT_PATH"

echo "Job submitted with name: $JOB_NAME"
echo "Output will be saved to: slurm_logs/${JOB_NAME}_*.out"
echo "Error will be saved to: slurm_logs/${JOB_NAME}_*.err"
