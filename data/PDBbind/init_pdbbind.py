import os
import tarfile
import pandas as pd
import sys
sys.path.append('../..')
from src.db_utils import db_connection
from pathlib import Path
from tqdm import tqdm

def create_pdbbind_table():
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pdbbind (
                id SERIAL PRIMARY KEY,
                pdb_id TEXT UNIQUE NOT NULL,
                binding_type TEXT,
                binding_value FLOAT,
                binding_unit TEXT,
                receptor_pdb BYTEA,
                ligand_sdf BYTEA,
                year INTEGER,
                resolution FLOAT,
                reference TEXT,
                ligand_name TEXT
            )
        """)
        conn.commit()

def parse_binding_data(binding_str):
    """Parse binding data string like 'Kd=49uM' into (type, value, unit)"""
    try:
        binding_type = binding_str[:binding_str.find('=')]
        value_unit = binding_str[binding_str.find('=')+1:]
        
        # Extract numeric value and unit
        for i, c in enumerate(value_unit):
            if not (c.isdigit() or c == '.' or c == '-'):
                value = float(value_unit[:i])
                unit = value_unit[i:]
                return binding_type, value, unit
    except:
        return None, None, None

def process_pdbbind_data(tar_path):
    print("Creating database table...")
    create_pdbbind_table()
    
    print("Opening tar file...")
    with tarfile.open(tar_path, 'r') as tar:
        # Read and parse the index file
        print("Reading index file...")
        index_file = tar.extractfile('refined-set/index/INDEX_refined_set.2020')
        index_content = index_file.read().decode('utf-8')
        
        # Skip header lines
        data_lines = [line.strip() for line in index_content.split('\n') if line.strip() and not line.startswith('#')]
        total_entries = len(data_lines)
        print(f"Found {total_entries} entries to process")
        
        with db_connection() as conn:
            cur = conn.cursor()
            
            # Process each entry with progress bar
            for line in tqdm(data_lines, desc="Processing PDBbind entries", unit="entry"):
                # Parse line
                parts = line.split()
                pdb_id = parts[0]
                resolution = float(parts[1])
                year = int(parts[2])
                binding_str = parts[3]
                reference = ' '.join(parts[4:])
                
                # Parse binding data
                binding_type, binding_value, binding_unit = parse_binding_data(binding_str)
                
                # Extract ligand name from reference
                ligand_name = reference.split('(')[-1].strip(')')
                
                try:
                    # Read receptor PDB and ligand SDF files from tar
                    receptor_pdb = tar.extractfile(f'refined-set/{pdb_id}/{pdb_id}_protein.pdb').read()
                    ligand_sdf = tar.extractfile(f'refined-set/{pdb_id}/{pdb_id}_ligand.sdf').read()
                    
                    # Insert into database
                    cur.execute("""
                        INSERT INTO pdbbind (
                            pdb_id, binding_type, binding_value, binding_unit,
                            receptor_pdb, ligand_sdf, year, resolution,
                            reference, ligand_name
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (pdb_id) DO UPDATE SET
                            binding_type = EXCLUDED.binding_type,
                            binding_value = EXCLUDED.binding_value,
                            binding_unit = EXCLUDED.binding_unit,
                            receptor_pdb = EXCLUDED.receptor_pdb,
                            ligand_sdf = EXCLUDED.ligand_sdf,
                            year = EXCLUDED.year,
                            resolution = EXCLUDED.resolution,
                            reference = EXCLUDED.reference,
                            ligand_name = EXCLUDED.ligand_name
                    """, (
                        pdb_id, binding_type, binding_value, binding_unit,
                        receptor_pdb, ligand_sdf, year, resolution,
                        reference, ligand_name
                    ))
                    conn.commit()
                except Exception as e:
                    print(f"\nError processing entry {pdb_id}: {str(e)}")
                    continue

if __name__ == "__main__":
    tar_path = os.path.expanduser("~/scratch/datasets/pdbbind/PDBbind_v2020_refined.tar")
    process_pdbbind_data(tar_path)



