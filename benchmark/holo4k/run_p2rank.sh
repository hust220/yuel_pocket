#!/bin/bash

# Load required Java module
module load java/21

# Get dataset folder from argument, default to "test340"
DATASET_FOLDER="${1:-test340}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
P2RANK_SCRIPT="$PROJECT_ROOT/baselines/p2rank_2.5.1/prank"

INPUT_DIR="$SCRIPT_DIR/$DATASET_FOLDER"
OUTPUT_DIR="$SCRIPT_DIR/${DATASET_FOLDER}_p2rank_predictions"
DS_FILE="$OUTPUT_DIR/dataset.ds"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found at $INPUT_DIR"
    exit 1
fi

# Check if p2rank script exists
if [ ! -f "$P2RANK_SCRIPT" ]; then
    echo "Error: p2rank script not found at $P2RANK_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create the dataset list file
echo "Creating dataset list from $INPUT_DIR..."
# P2Rank expects paths to PDB files.
# In generate_test_set.py we saved them as "{system_id}_protein.pdb"
find "$INPUT_DIR" -name "*_protein.pdb" > "$DS_FILE"

NUM_FILES=$(wc -l < "$DS_FILE")
if [ "$NUM_FILES" -eq 0 ]; then
    echo "No protein files found in $INPUT_DIR"
    exit 1
fi

echo "Found $NUM_FILES protein files."
echo "Running p2rank..."

# Run p2rank
# Use 'predict' command which is standard for p2rank
"$P2RANK_SCRIPT" predict "$DS_FILE" -o "$OUTPUT_DIR"

echo "P2Rank prediction finished. Results are in $OUTPUT_DIR"
