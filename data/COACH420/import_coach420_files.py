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
    """Create necessary tables for COACH420 dataset"""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Create proteins table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coach420_proteins (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                pdb TEXT NOT NULL,
                sequence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create ligands table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coach420_ligands (
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

def process_pdb_file(pdb_path):
    """Process a single PDB file and return protein and ligand data"""
    # Get protein name from filename (e.g., "148lE.pdb" -> "148lE")
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
                if residue.is_hetatm():
                    ligand_pdb = residue.to_pdb()
                    ligands.append((residue.res_name, ligand_pdb))
    
    return protein_name, sequence, pdb_content, ligands

def import_pdb_files(pdb_dir):
    """Import all PDB files from the specified directory"""
    pdb_dir = Path(pdb_dir)
    pdb_files = list(pdb_dir.glob('*.pdb'))
    
    with db_connection() as conn:
        cursor = conn.cursor()
        
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            try:
                protein_name, sequence, pdb_content, ligands = process_pdb_file(pdb_file)
                
                # Insert protein
                cursor.execute("""
                    INSERT INTO coach420_proteins (name, pdb, sequence)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO NOTHING
                """, (protein_name, pdb_content, sequence))
                
                # Insert ligands
                for ligand_name, ligand_pdb in ligands:
                    cursor.execute("""
                        INSERT INTO coach420_ligands (name, protein_name, pdb)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (name, protein_name) DO NOTHING
                    """, (ligand_name, protein_name, ligand_pdb))
                
                conn.commit()
                
            except Exception as e:
                print(f"Error processing {pdb_file.name}: {str(e)}")
                conn.rollback()
                raise e
                continue
        
        cursor.close()

if __name__ == '__main__':
    # Create database tables
    create_tables()
    
    # Import PDB files
    pdb_dir = os.path.expanduser('~/scratch/datasets/coach420')

    print(f"Importing PDB files from {pdb_dir}")

    import_pdb_files(pdb_dir)

# %%
