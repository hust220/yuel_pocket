#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PM_SCRIPT="$PROJECT_ROOT/baselines/gvp/src/pocket_miner.py"
PM_MODEL="$PROJECT_ROOT/baselines/gvp/models/pocketminer"

# Arguments
DATASET_FOLDER="${1:-test50}"
INPUT_DIR="$SCRIPT_DIR/$DATASET_FOLDER"
OUTPUT_DIR="$SCRIPT_DIR/${DATASET_FOLDER}_pocketminer_predictions"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found at $INPUT_DIR"
    exit 1
fi

# Check if PocketMiner script exists
if [ ! -f "$PM_SCRIPT" ]; then
    echo "Error: PocketMiner script not found at $PM_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Processing dataset: $DATASET_FOLDER"
echo "Output directory: $OUTPUT_DIR"

# Find all protein files
PROTEIN_FILES=($(find "$INPUT_DIR" -name "*_protein.pdb"))
TOTAL_FILES=${#PROTEIN_FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "No protein files found in $INPUT_DIR"
    exit 1
fi

echo "Found $TOTAL_FILES protein files."

# Create a list of tasks for batch prediction
TASK_LIST="$OUTPUT_DIR/tasks.txt"
rm -f "$TASK_LIST"

for pdb_file in "${PROTEIN_FILES[@]}"; do
    filename=$(basename "$pdb_file")
    base_name="${filename%_protein.pdb}"
    
    # We'll generate .pdb tasks for PocketMiner (B-factor info)
    output_pdb="$OUTPUT_DIR/${base_name}_predictions.pdb"
    echo "$pdb_file $output_pdb" >> "$TASK_LIST"
done

# Run PocketMiner in batch mode
echo "Running batch inference with PocketMiner..."
python3 "$PM_SCRIPT" --list "$TASK_LIST" --model "$PM_MODEL"

echo "PocketMiner prediction finished. Results are in $OUTPUT_DIR"
