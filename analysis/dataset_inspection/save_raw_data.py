#%%
import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add parent directory to path to import db_utils
sys.path.append('../..')
from src.db_utils import db_connection

def save_raw_data(pname, lname, output_dir):
    """Save raw data from database to files.
    
    Args:
        pname: protein name
        lname: ligand name
        output_dir: directory to save files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Query database for raw data
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT molecule_pos, molecule_one_hot, molecule_bonds,
                   protein_pos, protein_one_hot, protein_contacts,
                   protein_backbone, is_pocket
            FROM raw_datasets
            WHERE protein_name = %s AND ligand_name = %s
        """, (pname, lname))
        
        row = cursor.fetchone()
        if row is None:
            print(f"No data found for {pname} with {lname}")
            return
            
        # Unpack data and convert bytea to bytes
        (molecule_pos, molecule_one_hot, molecule_bonds,
         protein_pos, protein_one_hot, protein_contacts,
         protein_backbone, is_pocket) = [x.tobytes() if x is not None else None for x in row]
    
    # Save numpy arrays as text files
    arrays = {
        'molecule_pos': molecule_pos,
        'molecule_one_hot': molecule_one_hot,
        'molecule_bonds': molecule_bonds,
        'protein_pos': protein_pos,
        'protein_one_hot': protein_one_hot,
        'protein_contacts': protein_contacts,
        'protein_backbone': protein_backbone,
        'is_pocket': is_pocket
    }
    
    for name, data in arrays.items():
        if data is not None:
            # Load data using pickle
            arr = pickle.loads(data)
            # Save as text file with 2 decimal places and no scientific notation
            np.savetxt(os.path.join(output_dir, f'{name}.txt'), arr, fmt='%.2f')
    
    # Get and save PDB file
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pdb
            FROM proteins
            WHERE name = %s
        """, (pname,))
        pdb_data = cursor.fetchone()
        if pdb_data and pdb_data[0] is not None:
            with open(os.path.join(output_dir, f'{pname}.pdb'), 'wb') as f:
                f.write(pdb_data[0].tobytes())
    
    # Get and save MOL file
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT mol
            FROM ligands
            WHERE name = %s AND protein_name = %s
        """, (lname, pname))
        mol_data = cursor.fetchone()
        if mol_data and mol_data[0] is not None:
            with open(os.path.join(output_dir, f'{lname}.mol'), 'wb') as f:
                f.write(mol_data[0].tobytes())
    
    print(f"Data saved to {output_dir}")

def save_protein_structure(pname, output_dir):
    """Save protein structure data from database to files.
    
    Args:
        pname: protein name
        output_dir: directory to save files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Query database for protein structure data
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pdb, size, contacts
            FROM proteins
            WHERE name = %s
        """, (pname,))
        
        row = cursor.fetchone()
        if row is None:
            print(f"No protein structure data found for {pname}")
            return
            
        pdb_data, size, contacts_data = row
        
        # Save PDB file if available
        if pdb_data is not None:
            with open(os.path.join(output_dir, f'{pname}.pdb'), 'wb') as f:
                f.write(pdb_data.tobytes())
        
        # Save size information if available
        if size is not None:
            with open(os.path.join(output_dir, f'{pname}_size.txt'), 'w') as f:
                f.write(str(size))
        
        # Save contacts data if available
        if contacts_data is not None:
            # Load data using pickle
            arr = pickle.loads(contacts_data.tobytes())
            # Save as text file with 2 decimal places and no scientific notation
            np.savetxt(os.path.join(output_dir, f'{pname}_contacts.txt'), arr, fmt='%.2f')
    
    print(f"Protein structure data saved to {output_dir}")

if __name__ == '__main__':
    # pname = '3qdx'
    # lname = '3qdx_5'
    # pname = '1ugx'
    # lname = '1ugx_1'
    # pname = '1v6a'
    # lname = '1v6a_0'
    # pname = '1s3f'
    # lname = '1s3f_0'
    pname = '1d1v'
    lname = '1d1v_0'
    output_dir = f'raw_data_{pname}_{lname}'
    save_raw_data(pname, lname, output_dir) 

    # pname = '1vif'
    # pname = '4cw3'
    # output_dir = f'raw_data_{pname}'
    # save_protein_structure(pname, output_dir)
# %%
