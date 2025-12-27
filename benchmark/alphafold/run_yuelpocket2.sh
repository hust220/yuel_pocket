#!/bin/bash

# Arguments
DATASET_FOLDER="${1:-test1036_af_aligned}"
MODE="residues2" # Hardcoded to residues2 as requested
DEVICE="${2:-cuda}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
APP_SCRIPT="$PROJECT_ROOT/src/$MODE/app.py"

INPUT_DIR="$SCRIPT_DIR/$DATASET_FOLDER"
OUTPUT_DIR="$SCRIPT_DIR/${DATASET_FOLDER}_yuelpocket_${MODE}_predictions"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found at $INPUT_DIR"
    exit 1
fi

# Check if app script exists
if [ ! -f "$APP_SCRIPT" ]; then
    echo "Error: app script not found at $APP_SCRIPT (Mode: $MODE)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Processing dataset: $DATASET_FOLDER (Mode: $MODE)"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"

# Create a list of files first to count them
# AlphaFold dataset usually has *_protein.pdb or similar naming
PROTEIN_FILES=($(find "$INPUT_DIR" -maxdepth 1 -name "*_protein.pdb"))

# Fallback
if [ ${#PROTEIN_FILES[@]} -eq 0 ]; then
    PROTEIN_FILES=($(find "$INPUT_DIR" -maxdepth 1 -name "*.pdb" ! -name "*_ligand.pdb" ! -name "ligand.pdb"))
fi

TOTAL_FILES=${#PROTEIN_FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "No protein files found in $INPUT_DIR"
    exit 1
fi

echo "Found $TOTAL_FILES protein files."

# Generate a task list for batch prediction
TASK_LIST="$OUTPUT_DIR/tasks.txt"
rm -f "$TASK_LIST"

for pdb_file in "${PROTEIN_FILES[@]}"; do
    filename=$(basename "$pdb_file")
    if [[ "$filename" == *"_protein.pdb" ]]; then
        base_name="${filename%_protein.pdb}"
        ligand_files=($(find "$INPUT_DIR" \( -name "${base_name}_*_ligand.pdb" -o -name "${base_name}_ligand.sdf" -o -name "${base_name}_ligand.pdb" \)))
    else
        base_name="${filename%.pdb}"
        ligand_files=($(find "$INPUT_DIR" \( -name "${base_name}_ligand.pdb" -o -name "${base_name}_ligand.sdf" \)))
    fi
    
    if [ ${#ligand_files[@]} -eq 0 ]; then
        ligand_file="None"
    else
        ligand_file="${ligand_files[0]}"
    fi
    
    output_pdb="$OUTPUT_DIR/${base_name}_predictions.pdb"
    echo "$pdb_file $ligand_file $output_pdb" >> "$TASK_LIST"
done

NTASKS=$(wc -l < "$TASK_LIST")
if [ "$NTASKS" -eq 0 ]; then
    echo "No valid tasks generated."
    exit 1
fi

echo "Created task list with $NTASKS tasks at $TASK_LIST"
echo "Running batch inference..."

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python3 -m src.$MODE.app --list "$TASK_LIST" --device "$DEVICE" --cluster

echo "YuelPocket ($MODE) prediction finished. Results are in $OUTPUT_DIR"
