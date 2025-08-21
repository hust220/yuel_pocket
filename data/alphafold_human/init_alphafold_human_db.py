#%%

import os
import gzip
import tarfile
import pandas as pd
from pathlib import Path
import sys
sys.path.append('../../')
from src.db_utils import db_connection
from io import BytesIO
from tqdm import tqdm

def create_table(conn):
    """Create the alphafold_human table if it doesn't exist"""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alphafold_human (
                entry VARCHAR(20) PRIMARY KEY,
                reviewed BOOLEAN,
                entry_name VARCHAR(100),
                protein_names TEXT,
                gene_names TEXT,
                organism VARCHAR(200),
                length INTEGER,
                pdb TEXT
            )
        """)
    conn.commit()

def read_pdb_from_tar(tar, entry):
    """Read a single PDB file from tar archive"""
    pdb_filename = f'AF-{entry}-F1-model_v4.pdb.gz'
    try:
        member = tar.getmember(pdb_filename)
        f = tar.extractfile(member)
        if f is not None:
            with gzip.GzipFile(fileobj=BytesIO(f.read())) as gz:
                return gz.read().decode('utf-8')
    except (KeyError, tarfile.ReadError):
        return None
    return None

def main():
    # Paths
    base_dir = Path(os.path.expanduser('~/scratch/datasets/alphafold_human'))
    tsv_file = base_dir / 'uniprotkb_proteome_UP000005640_AND_revi_2025_06_06.tsv'
    tar_file = base_dir / 'UP000005640_9606_HUMAN_v4.tar'
    
    # Read TSV file
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Connect to database and create table
    with db_connection() as conn:
        create_table(conn)
        
        # Open tar file once
        with tarfile.open(tar_file, 'r:*') as tar:
            # Prepare data for insertion
            with conn.cursor() as cur:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing entries"):
                    entry = row['Entry']
                    pdb_content = read_pdb_from_tar(tar, entry)
                    
                    if pdb_content is not None:
                        cur.execute("""
                            INSERT INTO alphafold_human 
                            (entry, reviewed, entry_name, protein_names, gene_names, organism, length, pdb)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (entry) DO UPDATE SET
                            reviewed = EXCLUDED.reviewed,
                            entry_name = EXCLUDED.entry_name,
                            protein_names = EXCLUDED.protein_names,
                            gene_names = EXCLUDED.gene_names,
                            organism = EXCLUDED.organism,
                            length = EXCLUDED.length,
                            pdb = EXCLUDED.pdb
                        """, (
                            entry,
                            row['Reviewed'] == 'reviewed',
                            row['Entry Name'],
                            row['Protein names'],
                            row['Gene Names'],
                            row['Organism'],
                            row['Length'],
                            pdb_content
                        ))
        
        conn.commit()

if __name__ == '__main__':
    main()

# %%
