#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --account=dokhlab
#SBATCH --job-name=hard_decoys
#SBATCH --output=slurm_logs/hard_decoys_%a.out
#SBATCH --error=slurm_logs/hard_decoys_%a.err
#SBATCH --array=0-5

# Total number of chunks
NUM_CHUNKS=6

# Current chunk index from Slurm Array ID
CHUNK=$SLURM_ARRAY_TASK_ID

# Output path for this chunk
OUTPUT="hard_decoys_chunk_${CHUNK}.zip"

echo "Running task for chunk $CHUNK of $NUM_CHUNKS..."
mkdir -p slurm_logs

# Run the python script for the specific chunk
# Note: we assume we are running from the data/plinder directory
python compute_hard_decoys.py --num_chunks $NUM_CHUNKS --chunk $CHUNK --output $OUTPUT

echo "Chunk $CHUNK finished."
