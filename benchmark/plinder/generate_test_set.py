import os
import sys
import argparse
import random
from pathlib import Path
from zipfile import ZipFile
import pyarrow.parquet as pq
import pandas as pd
from rdkit import Chem

# Add project root to sys.path
PROJ_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(PROJ_ROOT)

# Data Paths
DATA_DIR = os.path.join(PROJ_ROOT, 'data', 'plinder', 'data', '2024-06', 'v2')
SYSTEMS_DIR = os.path.join(DATA_DIR, 'systems')
SPLIT_PATH = os.path.join(DATA_DIR, 'splits', 'split.parquet')

def get_split_ids(split='test', num_samples=50):
    print(f"Reading split file from {SPLIT_PATH} (Split: {split})...")
    try:
        table = pq.read_table(SPLIT_PATH, columns=['system_id', 'split'])
        df = table.to_pandas()
        split_df = df[df['split'] == split]
        all_ids = split_df['system_id'].tolist()
        
        if len(all_ids) < num_samples:
            print(f"Warning: Only found {len(all_ids)} {split} samples, requested {num_samples}")
            return all_ids
            
        return random.sample(all_ids, num_samples)
    except Exception as e:
        print(f"Error reading split file: {e}")
        return []

def extract_system(system_id, output_dir):
    # Logic adapted from src/residues/dataset.py
    zip_path = os.path.join(SYSTEMS_DIR, f"{system_id[1:3]}.zip")
    
    if not os.path.exists(zip_path):
        print(f"Zip not found for {system_id} at {zip_path}")
        return False

    output_dir = Path(output_dir)
    
    try:
        with ZipFile(zip_path, 'r') as zf:
            # 1. Extract Receptor
            receptor_path_in_zip = f"{system_id}/receptor.pdb"
            try:
                with zf.open(receptor_path_in_zip) as f:
                    receptor_pdb = f.read().decode("utf-8", "ignore")
                
                out_prot_path = output_dir / f"{system_id}_protein.pdb"
                with open(out_prot_path, 'w') as f:
                    f.write(receptor_pdb)
            except KeyError:
                print(f"Receptor not found for {system_id}")
                return False

            # 2. Extract Ligand (Logic from dataset.py to find best ligand)
            parts = system_id.split("__")
            ligand_ids = parts[3].split("_")
            
            candidate_ligands = []
            for lig_id in ligand_ids:
                sdf_path = f"{system_id}/ligand_files/{lig_id}.sdf"
                try:
                    with zf.open(sdf_path) as f:
                        sdf_content = f.read().decode("utf-8", "ignore")
                        candidate_ligands.append(sdf_content)
                except KeyError:
                    continue
            
            if not candidate_ligands:
                print(f"No ligands found for {system_id}")
                return False

            # Select largest
            best_sdf = None
            max_atoms = -1
            
            for sdf in candidate_ligands:
                mol = Chem.MolFromMolBlock(sdf, sanitize=False)
                if mol is not None:
                    n_atoms = mol.GetNumAtoms()
                    if n_atoms > max_atoms:
                        max_atoms = n_atoms
                        best_sdf = sdf
            
            if best_sdf:
                out_lig_path = output_dir / f"{system_id}_ligand.sdf"
                with open(out_lig_path, 'w') as f:
                    f.write(best_sdf)
                return True
            else:
                print(f"No valid ligand found for {system_id}")
                return False

    except Exception as e:
        print(f"Error extracting {system_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate random benchmark set from Plinder Splits")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of random samples to generate")
    parser.add_argument("--folder_name", type=str, default=None, help="Output folder name")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"], help="Which split to sample from")
    
    args = parser.parse_args()
    
    num_samples = args.num_samples
    folder_name = args.folder_name if args.folder_name else f"{args.split}{num_samples}"
    
    current_dir = Path(__file__).parent
    output_dir = current_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ids = get_split_ids(args.split, num_samples)
    if not ids:
        print("No IDs found. Exiting.")
        return

    print(f"Selected {len(ids)} random systems from {args.split} set.")
    
    success_count = 0
    for system_id in ids:
        if extract_system(system_id, output_dir):
            success_count += 1
            print(f"Extracted {system_id}")
            
    print(f"Successfully extracted {success_count}/{len(ids)} systems to {output_dir}")

if __name__ == "__main__":
    main()
