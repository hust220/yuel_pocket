#%%

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from db_utils import db_connection
import pandas as pd
from rdkit import Chem
import io

def extract_smiles_from_mol(mol_bytes):
    """Extract SMILES from mol bytea data"""
    try:
        # Convert bytea to bytes
        mol_data = bytes(mol_bytes)
        # Create RDKit molecule from mol data
        mol = Chem.MolFromMolBlock(mol_data.decode('utf-8'))
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return None
    except Exception as e:
        print(f"Error converting mol to SMILES: {e}")
        return None

def save_probes_to_csv(output_file='probes_data.csv'):
    """Save probe_set probes from ligands table to CSV file"""
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Query probe_set data joined with ligands table
        cur.execute("""
            SELECT l.name, l.mol, l.size 
            FROM probe_set p
            JOIN ligands l ON p.lname = l.name
            ORDER BY l.name
        """)
        
        results = cur.fetchall()
        
        # Process results
        data = []
        for name, mol_bytes, size in results:
            smiles = extract_smiles_from_mol(mol_bytes)
            if smiles is not None:
                data.append({
                    'PDB_ID': name,
                    'SMILES': smiles,
                    'size': size
                })
            else:
                print(f"Warning: Could not extract SMILES for {name}")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"Saved {len(df)} probes from probe_set to {output_file}")
        print(f"File saved at: {os.path.abspath(output_file)}")
        
        return df

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Save probes data
    output_file = os.path.join(plots_dir, "probes_data.csv")
    df = save_probes_to_csv(output_file)
    
    # Display summary
    print(f"\nSummary:")
    print(f"Total probes: {len(df)}")
    print(f"Size range: {df['size'].min()} - {df['size'].max()}")
    print(f"Average size: {df['size'].mean():.2f}")

# %%
