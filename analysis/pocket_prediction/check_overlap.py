#%%

import pandas as pd
from collections import Counter
import re
import sys
from pathlib import Path
sys.path.append('../../')
from src.db_utils import db_connection

def get_unique_proteins():
    with db_connection('yuel_pocket') as conn:
        cur = conn.cursor()
        
        # Get unique proteins from processed_datasets
        cur.execute("SELECT DISTINCT pname FROM processed_datasets;")
        processed_proteins = {row[0] for row in cur.fetchall()}
        
        # Get proteins from COACH420 and remove chain identifiers
        cur.execute("SELECT name FROM coach420_proteins;")
        coach420_proteins = {row[0][:-1] for row in cur.fetchall()}
        
        # Get proteins from Holo4K
        cur.execute("SELECT name FROM holo4k_proteins;")
        holo4k_proteins = {row[0] for row in cur.fetchall()}
        
        return processed_proteins, coach420_proteins, holo4k_proteins

def analyze_overlap():
    processed_proteins, coach420_proteins, holo4k_proteins = get_unique_proteins()
    
    # Calculate overlaps
    coach420_overlap = processed_proteins.intersection(coach420_proteins)
    holo4k_overlap = processed_proteins.intersection(holo4k_proteins)
    both_overlap = coach420_overlap.intersection(holo4k_overlap)
    
    # Print results
    print(f"Total unique proteins in processed datasets: {len(processed_proteins)}")
    print(f"Total proteins in COACH420 (after removing chain IDs): {len(coach420_proteins)}")
    print(f"Total proteins in Holo4K: {len(holo4k_proteins)}")
    print("\nOverlap Analysis:")
    print(f"Proteins in processed datasets that are in COACH420: {len(coach420_overlap)}")
    print(f"Proteins in processed datasets that are in Holo4K: {len(holo4k_overlap)}")
    print(f"Proteins in processed datasets that are in both COACH420 and Holo4K: {len(both_overlap)}")
    
    # Print some example overlapping proteins
    if coach420_overlap:
        print("\nExample proteins in processed datasets and COACH420:")
        print(list(coach420_overlap)[:5])
    
    if holo4k_overlap:
        print("\nExample proteins in processed datasets and Holo4K:")
        print(list(holo4k_overlap)[:5])

if __name__ == "__main__":
    analyze_overlap()

# %%
