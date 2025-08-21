import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
sys.path.append('../..')
from src.db_utils import db_connection
from src.lightning import YuelPocket
from yuel_pocket import _predict_pocket

def create_casf_prediction_table():
    """Create a table to store pocket prediction results for all target-ligand pairs"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS casf_predictions (
                id SERIAL PRIMARY KEY,
                target VARCHAR(4) NOT NULL,
                ligand VARCHAR(4) NOT NULL,
                pocket_pred BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(target, ligand)
            );
            CREATE INDEX IF NOT EXISTS idx_casf_pred_target ON casf_predictions(target);
            CREATE INDEX IF NOT EXISTS idx_casf_pred_ligand ON casf_predictions(ligand);
        """)
        conn.commit()

def get_unique_targets_and_ligands():
    """Get all unique targets and ligands from casf_screening table"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get unique targets
        cur.execute("SELECT DISTINCT target FROM casf_screening")
        targets = [row[0] for row in cur.fetchall()]
        
        # Get unique ligands
        cur.execute("SELECT DISTINCT ligand FROM casf_screening")
        ligands = [row[0] for row in cur.fetchall()]
        
        return targets, ligands

def predict_pockets():
    """Predict pockets for all target-ligand pairs"""
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '../../models/moad_bs8_date06-06_time11-19-53.016800/last.ckpt'
    model = YuelPocket.load_from_checkpoint(model_path, map_location=device).eval().to(device)
    
    # Get all targets and ligands
    targets, ligands = get_unique_targets_and_ligands()
    print(f"Found {len(targets)} targets and {len(ligands)} ligands")
    print(f"Total pairs to process: {len(targets) * len(ligands)}")
    
    # Create prediction table
    create_casf_prediction_table()
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Process each target-ligand pair
        total_pairs = len(targets) * len(ligands)
        with tqdm(total=total_pairs, desc="Processing target-ligand pairs") as pbar:
            for target in targets:
                # Get protein data from casf2016 table
                cur.execute("""
                    SELECT protein_pdb
                    FROM casf2016 
                    WHERE pdb_id = %s
                """, (target,))  # Add comma to make it a tuple
                target_result = cur.fetchone()
                protein_pdb = target_result[0].tobytes().decode('utf-8')
                
                for ligand in ligands:
                    try:
                        # Check if prediction already exists
                        cur.execute("""
                            SELECT id FROM casf_predictions 
                            WHERE target = %s AND ligand = %s
                        """, (target, ligand))
                        if cur.fetchone() is not None:
                            pbar.update(1)
                            continue
                        
                        # Get ligand data from casf2016 table
                        cur.execute("""
                            SELECT ligand_sdf
                            FROM casf2016
                            WHERE pdb_id = %s
                        """, (ligand,))  # Add comma to make it a tuple
                        ligand_result = cur.fetchone()
                        ligand_sdf = ligand_result[0].tobytes().decode('utf-8')
                        
                        # Run prediction
                        pocket_pred, _ = _predict_pocket(
                            protein_pdb,
                            ligand_sdf,
                            model,
                            distance_cutoff=10.0,
                            device=device
                        )
                        
                        # Convert predictions to numpy array
                        pocket_pred = pocket_pred.cpu().numpy().squeeze()
                        
                        # Store results
                        cur.execute("""
                            INSERT INTO casf_predictions (target, ligand, pocket_pred)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (target, ligand) DO UPDATE 
                            SET pocket_pred = EXCLUDED.pocket_pred
                        """, (target, ligand, pickle.dumps(pocket_pred)))
                        conn.commit()
                        
                    except Exception as e:
                        print(f"Error processing {target}-{ligand}: {str(e)}")
                        raise e
                    finally:
                        pbar.update(1)

if __name__ == "__main__":
    predict_pockets()
