#!/bin/bash
#SBATCH --account=nxd338_nih
#SBATCH --partition=mgc-nih
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32000MB
#SBATCH --time=240:00:00
#SBATCH --job-name=pocket
#SBATCH --output=job.out
#SBATCH --error=job.err

module load miniconda/3
conda activate torch

# Your commands here
echo "Starting job $SLURM_JOB_ID"
echo "Running on host $(hostname)"
echo "Using $SLURM_CPUS_ON_NODE CPUs"
echo "With GPU: $CUDA_VISIBLE_DEVICES"

python prepare_raw_dataset.py --num_workers 16 --batch_size 16

