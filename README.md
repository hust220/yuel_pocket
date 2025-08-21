# YuelPocket: Protein-ligand binding site prediction with graph neural network

YuelPocket is a deep learning model based on graph neural networks for predicting protein-ligand binding sites (pockets). The model takes protein structure and ligand information as input and predicts which residues are likely to form binding pockets.

## Environment Setup

Install the necessary packages:

```shell
pip install pdb-tools biopython imageio networkx rdkit
pip install torch torchvision lightning
pip install scipy scikit-learn tqdm wandb
```

## Model Preparation

Download the YuelPocket model:

```bash
wget https://zenodo.org/records/16921378/files/yuel_pocket.ckpt?download=1 -O models/yuel_pocket.ckpt
```

## Training

Train the model using the default MOAD configuration:

```bash
python train_yuel_pocket.py --config configs/train_moad.yml
```

### Configuration

Modify training parameters in `configs/train_moad.yml`:

```yaml
# Model parameters
lr: 2.0e-4          # Learning rate
batch_size: 8       # Batch size
n_layers: 16        # Number of GNN layers
n_epochs: 1000      # Training epochs
nf: 32              # Hidden dimension
activation: silu    # Activation function
```

### Training Parameters

Key command-line arguments:

- `--config`: Path to configuration file
- `--device`: Training device ('gpu' or 'cpu')
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--n_epochs`: Number of training epochs
- `--n_layers`: Number of GNN layers
- `--resume`: Resume training from checkpoint

## Prediction

### Single Prediction

**Using Python:**

```python
import torch
from src.lightning import YuelPocket
from yuel_pocket import predict_pocket

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YuelPocket.load_from_checkpoint('path/to/checkpoint.ckpt', map_location=device)

# Predict pockets
pocket_pred = predict_pocket(
    receptor_path='protein.pdb',
    ligand_path='ligand.sdf', 
    output_path='prediction.pdb',
    model=model,
    distance_cutoff=10.0,
    device=device
)
```

**Using Command Line:**

```bash
# Note: Command line interface is currently commented out in yuel_pocket.py
# To enable it, uncomment the argument parser section and main() call
python yuel_pocket.py protein.pdb ligand.sdf output.pdb --model models/yuel_pocket.ckpt --distance_cutoff 10.0
```

## Model Architecture

YuelPocket uses an E(n)-equivariant graph neural network (EGNN) architecture:

- **Input**: Protein residues (CA atoms) + ligand atoms + joint node
- **Node features**: Amino acid/atom one-hot encodings
- **Edge features**: Distance, backbone connectivity, protein-ligand interactions
- **Output**: Binary pocket prediction for each residue

Key hyperparameters:
- Hidden dimension: 32
- Number of layers: 16  
- Activation: SiLU
- Aggregation: Sum
- Distance cutoff: 10.0 Å

## Project Structure

```
yuel_pocket/
├── configs/           # Training configurations
├── data/             # Dataset initialization scripts
├── analysis/         # Analysis and evaluation scripts
├── src/              # Source code
│   ├── lightning.py  # PyTorch Lightning model
│   ├── gnn.py       # Graph neural network implementation
│   ├── datasets.py  # Dataset loading utilities
│   └── utils.py     # Utility functions
├── models/          # Saved model checkpoints
└── logs/            # Training logs
```

## Contact

If you have any questions, please contact me at jianopt@gmail.com
