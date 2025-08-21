import os
import tarfile
import io
import sys
from tqdm import tqdm
sys.path.append('../..')
from src.db_utils import db_connection

def create_casf2016_table():
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS casf2016 (
                id SERIAL PRIMARY KEY,
                pdb_id VARCHAR(4) NOT NULL,
                protein_pdb BYTEA,
                ligand_sdf BYTEA,
                pocket_pdb BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_casf2016_pdb_id ON casf2016(pdb_id);
        """)
        conn.commit()

def process_coreset(tar_path):
    with db_connection() as conn:
        cur = conn.cursor()
        
        with tarfile.open(tar_path, 'r') as tar:
            # Get all members in coreset directory
            coreset_members = [m for m in tar.getmembers() 
                             if 'CASF-2016/coreset/' in m.name 
                             and m.name.count('/') == 3]  # Only files in PDB directories
            
            # Group files by PDB ID
            pdb_files = {}
            for member in coreset_members:
                pdb_id = member.name.split('/')[-2]  # Get PDB ID from path
                if pdb_id not in pdb_files:
                    pdb_files[pdb_id] = {}
                
                if member.name.endswith('_protein.pdb'):
                    pdb_files[pdb_id]['protein_pdb'] = member
                elif member.name.endswith('_ligand.sdf'):
                    pdb_files[pdb_id]['ligand_sdf'] = member
                elif member.name.endswith('_pocket.pdb'):
                    pdb_files[pdb_id]['pocket_pdb'] = member
            
            # Process each PDB entry
            for pdb_id, files in tqdm(pdb_files.items(), desc="Processing PDB entries"):
                if not all(k in files for k in ['protein_pdb', 'ligand_sdf', 'pocket_pdb']):
                    print(f"Warning: Missing files for {pdb_id}, skipping...")
                    continue
                
                # Check if entry already exists
                cur.execute("SELECT id FROM casf2016 WHERE pdb_id = %s", (pdb_id,))
                if cur.fetchone() is not None:
                    print(f"Entry for {pdb_id} already exists, skipping...")
                    continue
                
                # Read files directly from tar
                protein_data = tar.extractfile(files['protein_pdb']).read()
                ligand_data = tar.extractfile(files['ligand_sdf']).read()
                pocket_data = tar.extractfile(files['pocket_pdb']).read()
                
                # Insert data
                cur.execute("""
                    INSERT INTO casf2016 (pdb_id, protein_pdb, ligand_sdf, pocket_pdb)
                    VALUES (%s, %s, %s, %s)
                """, (pdb_id, protein_data, ligand_data, pocket_data))
                
                conn.commit()

if __name__ == "__main__":
    tar_path = '/storage/home/juw1179/scratch/datasets/casf/CASF-2016.tar'
    
    # Create table
    create_casf2016_table()
    
    # Process and insert data
    process_coreset(tar_path)
    
    print("CASF-2016 database initialization completed!")

