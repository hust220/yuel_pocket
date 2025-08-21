#%%

import sys
import os
sys.path.append(os.path.join(__file__, '../../..'))

from src.db_utils import db_connection, add_column
from typing import List, Tuple, Set
from io import StringIO
from src.pdb_utils import Structure
from tqdm import tqdm

def parse_residue_id(residue: str) -> Tuple[str, int]:
    """
    Parse residue ID and return (chain_id, residue_number)
    
    Handle two formats:
    1. Allosteric site: 'A-PRO-150' -> ('A', 150)
    2. Active site: 'A-136' -> ('A', 136)
    """
    parts = residue.strip().split('-')
    chain_id = parts[0]
    
    try:
        if len(parts) == 3:  # Allosteric site format
            residue_number = int(parts[2])
        else:  # Active site format
            residue_number = int(parts[1])
        return chain_id, residue_number
    except:
        return None, None
        
def get_residues_with_ca(pdb_content: str) -> Set[Tuple[str, int]]:
    """
    Parse PDB content and return a set of residues (chain_id, residue_number) that have CA atoms
    """
    residues_with_ca = set()
    
    if not pdb_content:
        return residues_with_ca
    
    # Create Structure object from PDB content
    structure = Structure()
    structure.read(StringIO(pdb_content))
    
    # Iterate through all models, chains, residues
    for chain in structure[0]:
        for residue in chain:
            # Check if residue has CA atom
            ca_atom = residue.get_atom('CA')
            if ca_atom is not None:
                residues_with_ca.add((chain.chain_id, residue.res_id))
    
    return residues_with_ca

def create_residue_arrays(residue_list: List[str], residues_with_ca: Set[Tuple[str, int]]) -> List[int]:
    """
    Create a binary array (0/1) based on whether residues have CA atoms
    """
    result = []
    for residue in residues_with_ca:
        try:
            chain_id, res_num = parse_residue_id(residue)
            # Add 1 if residue has CA atom, 0 otherwise
            result.append(1 if (chain_id, res_num) in residues_with_ca else 0)
        except Exception as e:
            result.append(0)  # Default to 0 for error cases
    return result

def add_site_columns():
    """Add in_allosteric_site and in_active_site columns to allobench table"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Add new columns
        add_column('allobench', 'in_allosteric_site', 'integer[]')
        add_column('allobench', 'in_active_site', 'integer[]')
        
        # Get all records
        cur.execute("""
            SELECT id, pdb_id, allosteric_site_residues, active_site_residues, pdb_content 
            FROM allobench
        """)
        records = cur.fetchall()
        
        # Process records and collect data for batch update
        batch_data = []
        for record in tqdm(records, desc="Processing records"):
            id, pdb_id, allosteric_sites, active_sites, pdb_content = record
            # if pdb_id != '4CFE':
            #     continue

            # if len(allosteric_sites) == 0 and len(active_sites) == 0:
            #     continue

            # print(f"Processing record {id} with pdb_id {pdb_id}")
            allosteric_sites = [parse_residue_id(site) for site in allosteric_sites]
            active_sites = [parse_residue_id(site) for site in active_sites]
            
            # Get residues with CA atoms from PDB content
            residues_with_ca = get_residues_with_ca(pdb_content)
            
            in_allosteric = [1 if (chain_id, res_num) in allosteric_sites else 0 for chain_id, res_num in residues_with_ca]
            in_active = [1 if (chain_id, res_num) in active_sites else 0 for chain_id, res_num in residues_with_ca]
            
            batch_data.append((in_allosteric, in_active, id))
        
        # Perform batch update
        try:
            cur.executemany("""
                UPDATE allobench 
                SET in_allosteric_site = %s::integer[], 
                    in_active_site = %s::integer[]
                WHERE id = %s
            """, batch_data)
            conn.commit()
            print(f"Successfully updated {len(batch_data)} records")
            
        except Exception as e:
            conn.rollback()
            print(f"Error during batch update: {e}")

if __name__ == "__main__":
    add_site_columns() 
# %%
