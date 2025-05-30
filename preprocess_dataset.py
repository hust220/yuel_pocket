# according to the src/datasets.py, write a script to preprocess a raw dataset
# and save the processed dataset to a file

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from src.datasets import PocketDataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=str)
    parser.add_argument("pt_path", nargs='?', default=None, type=str)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    if args.pt_path is None:
        args.pt_path = args.pkl_path.replace(".pkl", ".pt")
    data_path = os.path.dirname(args.pkl_path)
    prefix = os.path.basename(args.pkl_path).split(".")[0]
    PocketDataset(data_path=data_path, prefix=prefix, noise=args.noise, device=device, save_path=args.pt_path)
