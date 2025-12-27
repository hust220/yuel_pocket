#!/bin/bash

# Arguments
DATASET_FOLDER="${1:-all}"
MODE="${2:-pos_sc3}" # Model mode: pos_aa3, pos_sc, etc.
MODEL_NAME="${3:-}" # Optional specific model checkpoint
DEVICE="${4:-cuda}"

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
if [ -n "$MODEL_NAME" ]; then
    echo "Using model: $MODEL_NAME"
    MODEL_ARG="--model $MODEL_NAME"
else
    echo "Using model: Default (from config)"
    MODEL_ARG=""
fi
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"

# Create a list of files first to count them
PROTEIN_FILES=($(find "$INPUT_DIR" -name "*_protein.pdb"))
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
    # In plinder, base_name is usually the system_id which ends before _protein.pdb
    base_name="${filename%_protein.pdb}" 
    
    # Try to find ligand as .pdb or .sdf
    ligand_files=($(find "$INPUT_DIR" \( -name "${base_name}_*_ligand.pdb" -o -name "${base_name}_ligand.sdf" -o -name "${base_name}_ligand.pdb" \)))
    
    if [ ${#ligand_files[@]} -eq 0 ]; then
        echo "Warning: No ligand file found for $pdb_file, skipping..."
        continue
    fi
    
    ligand_file="${ligand_files[0]}"
    output_txt="$OUTPUT_DIR/${filename}_predictions.txt"
    
    echo "$pdb_file $ligand_file $output_txt" >> "$TASK_LIST"
done

NTASKS=$(wc -l < "$TASK_LIST")
if [ "$NTASKS" -eq 0 ]; then
    echo "No valid tasks generated."
    exit 1
fi

echo "Created task list with $NTASKS pairs at $TASK_LIST"
echo "Running batch inference..."

cd "$PROJECT_ROOT" || exit 1
python -m src.$MODE.app --list "$TASK_LIST" $MODEL_ARG --save_pdb --device "$DEVICE" --cluster
cd "$SCRIPT_DIR" || exit 1

echo "YuelPocket ($MODE) prediction finished. Results are in $OUTPUT_DIR"
