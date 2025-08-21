#%%

import os
from pathlib import Path
import psycopg2
import sys
from tqdm import tqdm
sys.path.append('../../')
from src.db_utils import db_connection
from src.pdb_utils import Structure, three_to_one_letter

def create_tables():
    """Create necessary tables for HOLO4K dataset"""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Create proteins table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS holo4k_proteins (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                pdb TEXT NOT NULL,
                sequence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create ligands table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS holo4k_ligands (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                protein_name TEXT NOT NULL,
                pdb TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, protein_name)
            )
        """)
        
        conn.commit()
        cursor.close()

def get_protein_sequence(model):
    """Extract protein sequence from PDB structure"""
    sequence = []
    for chain in model:
        for residue in chain:
            if not residue.is_hetatm() and residue.res_name not in ['HOH', 'WAT', 'TIP3', 'H2O', 'SOL']:
                sequence.append(three_to_one_letter(residue.res_name))
    return ''.join(sequence)

def read_ligand_codes(ds_file):
    """Read ligand codes from .ds file"""
    ligand_codes = {}
    with open(ds_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('HEADER:'):
                parts = line.split()
                if len(parts) >= 2:
                    pdb_name = Path(parts[0]).stem
                    codes = parts[1].split(',')
                    ligand_codes[pdb_name] = codes
    return ligand_codes

def process_pdb_file(pdb_path, ligands):
    """Process a single PDB file and return protein and ligand data"""
    protein_name = Path(pdb_path).stem
        
    # Parse the structure
    structure = Structure()
    structure.read(pdb_path, skip_hetatm=False, skip_water=True)

    pdb_content = '\n'.join(r.to_pdb().rstrip() for c in structure[0] for r in c if not r.is_hetatm())
    
    # Extract protein sequence
    sequence = get_protein_sequence(structure[0])
    
    # Find ligands (HETATM records)
    ligands = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.res_name in ligands:
                    ligand_pdb = residue.to_pdb()
                    ligands.append((residue.res_name, ligand_pdb))
    
    return protein_name, sequence, pdb_content, ligands

def import_pdb_files(pdb_dir, ds_file):
    """Import all PDB files from the specified directory"""
    pdb_dir = Path(pdb_dir)
    pdb_files = list(pdb_dir.glob('*.pdb'))
    
    # Read ligand codes from .ds file
    ligand_codes = read_ligand_codes(ds_file)
    
    with db_connection() as conn:
        cursor = conn.cursor()
        
        nprocessed = 0
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            try:
                protein_name = Path(pdb_file).stem
                if protein_name not in ligand_codes:
                    # tqdm.write(f"Warning: No ligand codes found for {protein_name}")
                    continue
                protein_name, sequence, pdb_content, ligands = process_pdb_file(pdb_file, ligand_codes[protein_name])
                
                # Insert protein
                cursor.execute("""
                    INSERT INTO holo4k_proteins (name, pdb, sequence)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO NOTHING
                """, (protein_name, pdb_content, sequence))
                
                # Insert ligands
                for ligand_name, ligand_pdb in ligands:
                    cursor.execute("""
                        INSERT INTO holo4k_ligands (name, protein_name, pdb)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (name, protein_name) DO NOTHING
                    """, (ligand_name, protein_name, ligand_pdb))
                
                conn.commit()
                nprocessed += 1
            except Exception as e:
                print(f"Error processing {pdb_file.name}: {str(e)}")
                conn.rollback()
                raise e
                continue
        
        cursor.close()
    print(f'Total PDB files: {len(pdb_files)}')
    print(f'Processed PDB files: {nprocessed}')
    print(f'Skipped PDB files: {len(pdb_files) - nprocessed}')


if __name__ == '__main__':
    # Create database tables
    create_tables()
    
    # Import PDB files
    pdb_dir = os.path.expanduser('~/scratch/datasets/holo4k')
    ds_file = os.path.expanduser('~/scratch/datasets/holo4k/holo4k(mlig).ds')

    print(f"Importing PDB files from {pdb_dir}")
    print(f"Using ligand codes from {ds_file}")

    import_pdb_files(pdb_dir, ds_file)

# %%
