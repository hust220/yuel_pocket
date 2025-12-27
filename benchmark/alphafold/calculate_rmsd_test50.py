
import os
import warnings
from Bio.PDB import PDBParser, Superimposer, PDBIO
import numpy as np
import pandas as pd

# Suppress Bio.PDB warnings
warnings.filterwarnings("ignore")

test50_dir = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/test50"
test1036_dir = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/test1036"
output_csv = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/rmsd_test50.csv"

def get_ca_atoms(structure):
    atoms = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    # Residue ID tuple is (hetero_flag, sequence_identifier, insertion_code)
                    # We usually only care about hetero_flag == ' ' (standard residues)
                    # But sometimes modified residues are part of the protein.
                    # We'll calculate RMSD on all matching CA atoms.
                    res_id = residue.id
                    key = (chain.id, res_id)
                    atoms[key] = residue['CA']
        break # Only first model
    return atoms

results = []

print(f"Calculating RMSD for files in {test50_dir} against {test1036_dir}...")

if not os.path.exists(test50_dir):
    print(f"Error: {test50_dir} does not exist")
    exit(1)

files = sorted([f for f in os.listdir(test50_dir) if f.endswith("_protein.pdb")])

parser = PDBParser(QUIET=True)

for f in files:
    system_id = f.replace("_protein.pdb", "")
    p50_path = os.path.join(test50_dir, f)
    p1036_path = os.path.join(test1036_dir, f)
    
    if not os.path.exists(p1036_path):
        print(f"Skipping {system_id}: Reference file not found in test1036")
        continue
        
    try:
        struct50 = parser.get_structure("test50", p50_path)
        struct1036 = parser.get_structure("test1036", p1036_path)
        
        atoms50_dict = get_ca_atoms(struct50)
        atoms1036_dict = get_ca_atoms(struct1036)
        
        # Find common keys
        common_keys = set(atoms50_dict.keys()) & set(atoms1036_dict.keys())
        
        if len(common_keys) < 3:
            print(f"Skipping {system_id}: Too few common residues ({len(common_keys)})")
            continue
            
        # Create lists of atoms for superposition
        # We align 50 to 1036 (reference)
        fixed_atoms = []
        moving_atoms = []
        
        # Sort keys to ensure deterministic order
        sorted_keys = sorted(list(common_keys), key=lambda x: (x[0], x[1][1], x[1][2]))
        
        for k in sorted_keys:
            fixed_atoms.append(atoms1036_dict[k])
            moving_atoms.append(atoms50_dict[k])
            
        sup = Superimposer()
        sup.set_atoms(fixed_atoms, moving_atoms)
        # We don't necessarily need to transform the structure if we just want the RMSD value from the alignment
        # sup.rms property contains the RMSD of the aligned atoms after superposition
        
        rmsd = sup.rms
        
        results.append({
            "SystemID": system_id,
            "RMSD": rmsd,
            "CommonResidues": len(common_keys)
        })
        print(f"Processing {system_id}: RMSD = {rmsd:.4f} ({len(common_keys)} residues)")
        
    except Exception as e:
        print(f"Error processing {system_id}: {e}")

if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nDone. Results saved to {output_csv}")
    print(f"Average RMSD: {df['RMSD'].mean():.4f}")
    print(f"Median RMSD: {df['RMSD'].median():.4f}")
    print(f"Min RMSD: {df['RMSD'].min():.4f}")
    print(f"Max RMSD: {df['RMSD'].max():.4f}")
else:
    print("No results calculated.")
