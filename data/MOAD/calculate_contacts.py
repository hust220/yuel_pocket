import argparse
import pickle
from multiprocessing import Pool
from functools import partial
from rdkit import Chem
from tqdm import tqdm
import psycopg2
from io import StringIO, BytesIO
import sys
import numpy as np
import subprocess
import os
import tempfile
import time
sys.path.append('../../')
from src import const

DB_ARGS = {
    'dbname': 'moad',
    'user': 'juw1179',
    'host': 'submit03',
    'port': 5433
}

# try reconnecting if the connection is failed, and retry 5 times, wait time double each time
def db_connect(db_args):
    for i in range(5):
        try:
            return psycopg2.connect(**db_args)
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            time.sleep(2**i)
    raise Exception("Failed to connect to database")

def add_contacts_column():
    """Add contacts column to proteins table if it doesn't exist"""
    conn = db_connect(DB_ARGS)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            ALTER TABLE proteins 
            ADD COLUMN IF NOT EXISTS contacts bytea
        """)
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"Error adding contacts column: {str(e)}")
        conn.rollback()
    finally:
        if conn:
            conn.close()

def get_proteins_to_process():
    """Get all proteins that don't have contacts calculated yet"""
    conn = db_connect(DB_ARGS)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM proteins 
            WHERE contacts IS NULL
        """)
        proteins = cursor.fetchall()
        cursor.close()
    except Exception as e:
        print(f"Error getting proteins to process: {str(e)}")
        conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()
        return proteins if proteins else []

def get_pdb_data(protein_name):
    """Get PDB data for a protein"""
    conn = db_connect(DB_ARGS)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pdb FROM proteins 
            WHERE name = %s
        """, (protein_name,))
        pdb_data = cursor.fetchone()
        cursor.close()
    except Exception as e:
        print(f"Error getting PDB data for {protein_name}: {str(e)}")
        conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()
        return pdb_data[0].tobytes() if pdb_data else None

def calculate_contacts(protein_name, pdb_data):
    """Calculate contacts for a protein using external script"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as pdb_file, \
             tempfile.NamedTemporaryFile(suffix='.contacts', delete=False) as contacts_file:
            
            # Write PDB data to file
            pdb_file.write(pdb_data)
            pdb_file.flush()
            
            # Run contacts calculation script
            script_path = os.path.join(os.path.dirname(__file__), 'contacts.sh')
            subprocess.run(
                ["bash", script_path, pdb_file.name, contacts_file.name],
                check=True
            )
            
            # Read contacts file as binary
            with open(contacts_file.name, 'rb') as f:
                contacts_binary = f.read()
            
            # Clean up
            os.unlink(pdb_file.name)
            os.unlink(contacts_file.name)
            
            return contacts_binary
            
    except Exception as e:
        print(f"Error calculating contacts for {protein_name}: {str(e)}")
        return None

def update_protein_contacts(protein_name, contacts_binary):
    """Update protein record with contacts data"""
    conn = db_connect(DB_ARGS)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE proteins 
            SET contacts = %s 
            WHERE name = %s
        """, (psycopg2.Binary(contacts_binary), protein_name))
        conn.commit()
    except Exception as e:
        print(f"Error updating contacts for {protein_name}: {str(e)}")
        conn.rollback()
    finally:
        if conn:
            conn.close()
        cursor.close()

def process_protein(protein_name):
    """Process a single protein to calculate and store contacts"""
    pdb_data = get_pdb_data(protein_name)
    contacts_binary = calculate_contacts(protein_name, pdb_data)
    if contacts_binary:
        update_protein_contacts(protein_name, contacts_binary)
    return protein_name

def main(args):
    # Ensure contacts column exists
    add_contacts_column()
    
    # Get proteins to process
    proteins = get_proteins_to_process()
    print(f"Found {len(proteins)} proteins to process")
    
    if args.max_proteins:
        proteins = proteins[:args.max_proteins]
        print(f"Processing first {len(proteins)} proteins")
    
    # Process proteins in parallel
    with Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_protein, proteins),
            total=len(proteins),
            desc="Processing proteins"
        ))
    
    print(f"Completed processing {len(results)} proteins")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8, 
                       help='Number of worker processes')
    parser.add_argument('--max_proteins', type=int, default=None,
                       help='Maximum number of proteins to process')
    args = parser.parse_args()
    
    main(args)
