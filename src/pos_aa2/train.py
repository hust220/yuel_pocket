import argparse

from .model import YuelPocket
from .dataset import PocketDataset
from .config import get_config
from ..utils import disable_rdkit_logging, run_training

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train YuelBond2D model')
    p.add_argument('--mode', type=str, default='plinder', choices=['plinder'],
                   help='Configuration mode: plinder')

    disable_rdkit_logging()

    args = p.parse_args()
    
    # Load config from config.py
    config = get_config(args.mode)
    
    # Add project name for wandb
    config['project'] = 'yuel_pocket'
    
    # Add dataset_class to config
    config['dataset_class'] = PocketDataset
    
    # Run training using the generic run_training function
    run_training(args=config, model=YuelPocket, dataset=PocketDataset)

