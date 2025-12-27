import os
import sys
import numpy as np
import pyarrow.parquet as pq
from zipfile import ZipFile
from rdkit import Chem
import random

# Path configuration matches src/residues/dataset.py
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, 'data', 'plinder', 'data', '2024-06', 'v2')
SYSTEMS_DIR = os.path.join(DATA_DIR, 'systems')
SPLIT_PATH = os.path.join(DATA_DIR, 'splits', 'split.parquet')

def get_random_test_system():
    print(f"Reading split file from {SPLIT_PATH}...")
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Split file not found at {SPLIT_PATH}")
        
    table = pq.read_table(SPLIT_PATH, columns=['system_id', 'split'])
    df = table.to_pandas()
    
    # Filter for test split
    test_df = df[df['split'] == 'test']
    if test_df.empty:
        raise ValueError("No test entries found in split file.")
        
    print(f"Found {len(test_df)} test systems.")
    
    # Pick random system
    random_row = test_df.sample(n=1).iloc[0]
    system_id = random_row['system_id']
    print(f"Selected system: {system_id}")
    
    return system_id

def extract_system_files(system_id, output_dir):
    # Determine zip path logic from dataset.py: zip_path = os.path.join(SYSTEMS_DIR, f"{system_id[1:3]}.zip")
    bucket = system_id[1:3]
    zip_path = os.path.join(SYSTEMS_DIR, f"{bucket}.zip")
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip bucket {zip_path} not found for {system_id}")
        
    print(f"Reading from {zip_path}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with ZipFile(zip_path, 'r') as zf:
        # 1. Extract Receptor
        receptor_zip_path = f"{system_id}/receptor.pdb"
        try:
            with zf.open(receptor_zip_path) as f:
                receptor_content = f.read()
        except KeyError:
             print(f"Error: {receptor_zip_path} not found in zip.")
             return False

        receptor_out_path = os.path.join(output_dir, "receptor.pdb")
        with open(receptor_out_path, "wb") as f:
            f.write(receptor_content)
        print(f"Saved receptor to {receptor_out_path}")

        # 2. Extract Ligand
        # Logic from dataset.py to find ligands
        parts = system_id.split("__")
        # Format seems to be PDB__...__...__LIGANDS__...
        if len(parts) > 3:
            ligands_str = parts[3]
            ligand_ids = ligands_str.split("_")
        else:
            # Fallback for unexpected ID format, try listing files in zip
            print("Warning: unexpected system_id format, scanning zip for ligands...")
            file_list = zf.namelist()
            ligand_prefix = f"{system_id}/ligand_files/"
            possible_files = [f for f in file_list if f.startswith(ligand_prefix) and f.endswith(".sdf")]
            if not possible_files:
                print("No ligand files found in zip.")
                return False
            # Just pick the first one from list if we can't parse ID
            ligand_ids = [os.path.basename(f).replace('.sdf', '') for f in possible_files]

        # Select best ligand (largest by atoms, as per dataset.py)
        best_mol_content = None
        max_atoms = -1
        best_lig_id = None
        
        for lig_id in ligand_ids:
            sdf_zip_path = f"{system_id}/ligand_files/{lig_id}.sdf"
            try:
                with zf.open(sdf_zip_path) as f:
                    content = f.read()
                    mol_block = content.decode("utf-8", "ignore")
                    
                    mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
                    if mol is not None:
                        n_atoms = mol.GetNumAtoms()
                        if n_atoms > max_atoms:
                            max_atoms = n_atoms
                            best_mol_content = content
                            best_lig_id = lig_id
            except KeyError:
                continue
        
        if best_mol_content is None:
            print("Failed to load any valid ligands.")
            return False

        ligand_out_path = os.path.join(output_dir, "ligand.sdf")
        with open(ligand_out_path, "wb") as f:
            f.write(best_mol_content)
        print(f"Saved best ligand ({best_lig_id}, {max_atoms} atoms) to {ligand_out_path}")
        
        return True

def main():
    try:
        system_id = get_random_test_system()
        out_dir = os.path.join(PROJ_ROOT, "tests", "test_case")
        
        success = extract_system_files(system_id, out_dir)
        
        if success:
            print("\nSuccess!")
            print(f"Run the following command to test:")
            print(f"python -m src.residues.app --pdb {os.path.join(out_dir, 'receptor.pdb')} --ligand {os.path.join(out_dir, 'ligand.sdf')} --output {os.path.join(out_dir, 'out.txt')}")
        else:
            print("Extraction failed.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
