#!/bin/bash

# Parameters
#SBATCH --account=nxd338_nih
#SBATCH --error=/scratch/juw1179/codes/yuel_pocket/analysis/probe_set/probe_jobs/%j_0_log.err
#SBATCH --gpus-per-task=1
#SBATCH --job-name=probe_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/juw1179/codes/yuel_pocket/analysis/probe_set/probe_jobs/%j_0_log.out
#SBATCH --partition=mgc-nih
#SBATCH --signal=USR2@90
#SBATCH --time=48:00:00
#SBATCH --wckey=submitit

# setup
cd /scratch/juw1179/codes/yuel_pocket/analysis/probe_set
module load miniconda/3
conda activate torch
export PYTHONPATH=/scratch/juw1179/codes/yuel_pocket:$PYTHONPATH

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /scratch/juw1179/codes/yuel_pocket/analysis/probe_set/probe_jobs/%j_%t_log.out --error /scratch/juw1179/codes/yuel_pocket/analysis/probe_set/probe_jobs/%j_%t_log.err --cpu-bind=none /storage/work/juw1179/.conda/envs/torch/bin/python -u -m submitit.core._submit /scratch/juw1179/codes/yuel_pocket/analysis/probe_set/probe_jobs
