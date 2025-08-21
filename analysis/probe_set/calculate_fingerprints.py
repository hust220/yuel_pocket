import os
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../../')
from src.db_utils import db_connection
import pickle

def create_fingerprint_table():
    """Create a table to store protein fingerprints based on probe predictions."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # First get all probes to create columns
        cur.execute("SELECT DISTINCT probe_name FROM probe_moad_predictions ORDER BY probe_name")
        probes = [row[0] for row in cur.fetchall()]
        
        # Create table if not exists with both fingerprint and individual probe columns
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS probe_fingerprints (
                id SERIAL PRIMARY KEY,
                pname TEXT NOT NULL UNIQUE,
                fingerprint BYTEA NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                {}
            );
            CREATE INDEX IF NOT EXISTS idx_probe_fingerprints_pname ON probe_fingerprints(pname);
        """.format(
            ',\n                '.join([f'probe_{probe.replace("-", "_")} FLOAT' for probe in probes])
        )
        
        cur.execute(create_table_sql)
        conn.commit()

def get_all_proteins():
    """Get all protein names that have probe predictions."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT pname FROM probe_moad_predictions")
        return [row[0] for row in cur.fetchall()]

def get_all_probes():
    """Get all probe names in a fixed order."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT probe_name FROM probe_moad_predictions ORDER BY probe_name")
        return [row[0] for row in cur.fetchall()]

def get_all_probe_predictions(pname):
    """Get pocket predictions for all probes of a specific protein."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT probe_name, pocket_pred
            FROM probe_moad_predictions 
            WHERE pname = %s
        """, (pname,))
        
        results = {}
        for row in cur.fetchall():
            probe_name, pocket_pred = row
            results[probe_name] = pickle.loads(pocket_pred)
        
        return results

def calculate_protein_fingerprints():
    """
    Calculate fingerprints for all proteins based on probe predictions.
    Stores both a binary fingerprint array and individual probe columns.
    """
    # Create fingerprint table
    create_fingerprint_table()
    
    # Get all proteins and probes
    proteins = get_all_proteins()
    probes = get_all_probes()
    
    print(f"Found {len(proteins)} proteins and {len(probes)} probes")
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Process each protein
        for pname in tqdm(proteins, desc="Processing proteins"):
            try:
                # Check if fingerprint already exists
                cur.execute("""
                    SELECT id FROM probe_fingerprints 
                    WHERE pname = %s
                """, (pname,))
                if cur.fetchone() is not None:
                    continue
                
                # Initialize fingerprint array and probe values dictionary
                fingerprint = np.zeros(len(probes), dtype=np.float32)
                probe_values = {}
                
                # Get all probe predictions for this protein at once
                all_predictions = get_all_probe_predictions(pname)
                
                # Calculate maximum prediction values for each probe
                for i, probe in enumerate(probes):
                    if probe in all_predictions:
                        max_val = float(np.max(all_predictions[probe]))
                        fingerprint[i] = max_val
                        probe_values[f'probe_{probe.replace("-", "_")}'] = max_val
                    else:
                        probe_values[f'probe_{probe.replace("-", "_")}'] = 0.0
                
                # Prepare SQL for inserting values
                columns = ['pname', 'fingerprint'] + list(probe_values.keys())
                values = [pname, pickle.dumps(fingerprint)] + list(probe_values.values())
                placeholders = ['%s'] * len(values)
                
                # Create the INSERT query
                insert_sql = f"""
                    INSERT INTO probe_fingerprints ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT (pname) DO UPDATE SET 
                    fingerprint = EXCLUDED.fingerprint,
                    {', '.join(f"{col} = EXCLUDED.{col}" for col in probe_values.keys())}
                """
                
                # Execute the insert
                cur.execute(insert_sql, values)
                conn.commit()
                
            except Exception as e:
                print(f"Error processing {pname}: {str(e)}")
                continue

if __name__ == "__main__":
    calculate_protein_fingerprints() 