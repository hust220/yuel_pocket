#%%

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
from src.db_utils import db_connection

def add_size_column():
    """Add size column to proteins table if it doesn't exist"""
    with db_connection('moad') as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                ALTER TABLE proteins 
                ADD COLUMN IF NOT EXISTS size integer
            """)
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error adding size column: {str(e)}")
            conn.rollback()

def get_proteins_to_process():
    """Get all proteins that don't have size calculated yet"""
    with db_connection('moad') as conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM proteins 
                WHERE size IS NULL
            """)
            proteins = cursor.fetchall()
            cursor.close()
            return proteins if proteins else []
        except Exception as e:
            print(f"Error getting proteins to process: {str(e)}")
            conn.rollback()
            raise e

def get_pdb_data(protein_name):
    """Get PDB data for a protein"""
    with db_connection('moad') as conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pdb FROM proteins 
                WHERE name = %s
            """, (protein_name,))
            pdb_data = cursor.fetchone()
            cursor.close()
            return pdb_data[0].tobytes() if pdb_data else None
        except Exception as e:
            print(f"Error getting PDB data for {protein_name}: {str(e)}")
            conn.rollback()
            raise e

def calculate_protein_size(protein_name, pdb_data):
    """Calculate size of a protein by counting number of unique residues using chain, number, and name from the first model only"""
    try:
        # Process PDB data directly from memory
        residues = set()
        with BytesIO(pdb_data) as pdb_stream:
            for line in pdb_stream:
                line = line.decode('utf-8')
                if line.startswith('ENDMDL'):
                    # Stop after first model
                    break
                if line.startswith('ATOM'):
                    # Chain ID is in column 21
                    chain = line[21]
                    # Residue sequence number is in columns 22-26
                    residue_num = line[22:26].strip()
                    # Residue name is in columns 17-20
                    residue_name = line[17:20].strip()
                    # Create unique key combining chain, number and name
                    residue_key = f"{chain}_{residue_num}_{residue_name}"
                    residues.add(residue_key)
        
        return len(residues)
            
    except Exception as e:
        print(f"Error calculating size for {protein_name}: {str(e)}")
        return None

def update_protein_size(protein_name, size):
    """Update protein record with size data"""
    with db_connection('moad') as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE proteins 
                SET size = %s 
                WHERE name = %s
            """, (size, protein_name))
            conn.commit()
        except Exception as e:
            print(f"Error updating size for {protein_name}: {str(e)}")
            conn.rollback()
        finally:
            cursor.close()

def process_protein(protein_name):
    """Process a single protein to calculate and store size"""
    pdb_data = get_pdb_data(protein_name)
    size = calculate_protein_size(protein_name, pdb_data)
    if size is not None:
        update_protein_size(protein_name, size)
    return protein_name

def main(args):
    # Ensure size column exists
    add_size_column()
    
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
