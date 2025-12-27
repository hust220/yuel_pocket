# YuelPocket: Protein-ligand binding site prediction with graph neural network

YuelPocket is a deep learning model based on graph neural networks for predicting protein-ligand binding sites. The model operates in two distinct modes:

1.  **Residue-Level Mode (`residues2`)**: Predicts which protein residues are part of the binding pocket.
2.  **Coordinate-Level Mode (`pos_sc3`)**: Predicts precise binding site coordinates using SAS (Solvent Accessible Surface) points as probes.

## Environment Setup

Install the necessary packages:

```shell
pip install pdb-tools biopython imageio networkx rdkit pyarrow pandas
pip install torch torchvision lightning
pip install scipy scikit-learn tqdm wandb
pip install egnn-pytorch
```

> **Important**: Always run the python scripts as modules (e.g., `python -m src.residues2.app`) from the project root directory. Do **not** run them directly by path (e.g., `python src/residues2/app.py`), as this will cause import errors.

## Modes & Pretrained Models

You can find pretrained checkpoints in the `checkpoints/` directory.

### 1. Residue-Level Mode (`residues2`)

This mode predicts the probability of each residue belonging to a pocket.

- **Source Code**: `src/residues2/`
- **Checkpoint**: `checkpoints/plinder_residues2.ckpt`
- **Inference**:


```bash
python -m src.residues2.app \
    examples/receptor.pdb \
    examples/ligand.sdf \
    results/residues.txt \
    --model checkpoints/plinder_residues2.ckpt
```

**Batch Processing:**
```bash
python -m src.residues2.app --list tasks.txt --model checkpoints/plinder_residues2.ckpt
```
*`tasks.txt` format per line: `receptor.pdb ligand.sdf output.txt`*

**Arguments:**
- `pdb`: Path to the receptor PDB file.
- `ligand`: Path to the ligand file (SDF, MOL, or PDB).
- `output`: Path for the output file. 
    - Always generates a `.txt` file with residue probabilities.
    - If `output` ends with `.pdb`, an annotated PDB file (scores in B-factor column) is also generated.
- `--list`: File containing a list of inputs for batch processing.
- `--model`: Path to the model checkpoint.
- `--device`: `cpu` or `cuda` (default: `cpu`).
- `--cluster`: Enable clustering of SAS points (optional).

### 2. Coordinate-Level Mode (`pos_sc3`)

This mode predicts pocket centers by scoring probe points sampled from the protein surface.

- **Source Code**: `src/pos_sc3/`
- **Checkpoint**: `checkpoints/plinder_pos_sc3.ckpt`
- **Inference**:

```bash
python -m src.pos_sc3.app \
    examples/receptor.pdb \
    examples/ligand.sdf \
    results/coordinates.txt \
    --model checkpoints/plinder_pos_sc3.ckpt \
    --cluster
```

**Batch Processing:**
```bash
python -m src.pos_sc3.app --list tasks.txt --model checkpoints/plinder_pos_sc3.ckpt --cluster
```

**Arguments:**
- `pdb`: Path to the receptor PDB file.
- `ligand`: Path to the ligand file.
- `output`: Path for the output TXT file (contains coordinate scores).
- `--list`: Batch processing list file.
- `--model`: Path to the model checkpoint.
- `--save_pdb`: Save SAS points as a PDB file.
- `--cluster`: Enable clustering to find pocket centers.
- `--k_nn`: Number of neighbors for clustering (default: 10).


## Training

To train the models from scratch using the PLINDER dataset, you first need to prepare the data.

### Weights & Biases Logging

The training scripts use [Weights & Biases](https://wandb.ai/) for tracking experiment metrics. To enable logging:

1.  Create a file named `wandb_api_key.txt` in the root directory.
2.  Paste your Wandb API key into this file.

If the file is not found, the script will show a warning but proceed with training (ensure you are logged in manually via `wandb login` if you want to track runs).

### Data Preparation

Scripts for downloading and processing the PLINDER dataset are available in `scripts/plinder_prep/`.

1.  **Download Data**:
    Use `download_plinder.sh` to download the dataset. The models read directly from the PLINDER zip structure.

2.  **Precompute SAS Points**:
    Run `precompute_sas.py` to generate the surface points required for the `pos_sc3` model.

### Training Commands

**Train Residue-Level Model:**
```bash
python -m src.residues2.train
```

**Train Coordinate-Level Model:**
```bash
python -m src.pos_sc3.train
```

*Note: Adjust hyperparameters in the respective `config.py` files.*




## Project Structure

```
yuel_pocket/
├── baselines/        # Baseline methods (P2Rank, GVP, etc.)
├── benchmark/        # Benchmarking scripts (Holo4k, PDBbind, PLINDER)
├── checkpoints/      # Pretrained model weights
├── src/              # Source code
│   ├── residues2/    # Residue-level prediction module
│   ├── pos_sc3/      # Coordinate-level prediction module
│   ├── egnn.py       # EGNN implementation
│   ├── pdb_utils.py  # PDB processing utilities
│   └── ...
└── analysis/         # Analysis scripts
```

## Contact

If you have any questions, please contact me at jianopt@gmail.com
