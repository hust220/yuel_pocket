import sys
import os
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
# Add the src directory to the Python path
src_dir = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(src_dir))

import db_utils

def add_all_pockets_column():
    """Add all_pockets column to proteins table if it doesn't exist"""
    db_utils.add_column('proteins', 'all_pockets', 'bytea')

def combine_pockets():
    """Combine is_pocket data from raw_datasets for each protein"""
    with db_utils.db_connection() as conn:
        cur = conn.cursor()
        
        # Get all unique protein names
        cur.execute("""
            SELECT DISTINCT protein_name 
            FROM raw_datasets 
            WHERE protein_name IS NOT NULL
        """)
        protein_names = [row[0] for row in cur.fetchall()]
        
        # Add progress bar for proteins
        for protein_name in tqdm(protein_names, desc="Processing proteins"):
            # Get all is_pocket data for this protein
            cur.execute("""
                SELECT is_pocket 
                FROM raw_datasets 
                WHERE protein_name = %s 
                AND is_pocket IS NOT NULL
            """, (protein_name,))
            
            pocket_data = []
            rows = cur.fetchall()
            
            # Add progress bar for pocket data loading
            for row in tqdm(rows, desc=f"Loading pockets for {protein_name}", leave=False):
                pocket_array = pickle.loads(row[0])
                pocket_data.append(pocket_array)
            
            if not pocket_data:
                print(f"No pocket data found for protein: {protein_name}")
                continue
            
            # Combine pockets using logical OR
            combined_pockets = np.logical_or.reduce(pocket_data)
            
            # Convert to bytes for storage
            combined_pockets_bytes = pickle.dumps(combined_pockets)
            
            # Update proteins table
            cur.execute("""
                UPDATE proteins 
                SET all_pockets = %s 
                WHERE name = %s
            """, (combined_pockets_bytes, protein_name))
            
            conn.commit()
            # print(f"Updated all_pockets for protein: {protein_name}")

if __name__ == "__main__":
    add_all_pockets_column()
    combine_pockets()