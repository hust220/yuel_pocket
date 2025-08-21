import os
import sys
import torch
import numpy as np
from tqdm import tqdm
sys.path.append('../../')
from src.db_utils import db_connection
import pickle
from src.lightning import YuelPocket
from src.datasets import collate
from yuel_pocket import PdbSdfDataset

def create_results_table():
    """Create table to store pocket predictions for all protein-probe pairs."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS probe_moad_test (
                protein_name TEXT NOT NULL,
                probe_name TEXT NOT NULL,
                pocket_pred BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (protein_name, probe_name)
            );
            
            -- Create indices for faster querying
            CREATE INDEX IF NOT EXISTS idx_probe_moad_test_pred_protein 
            ON probe_moad_test(protein_name);
            
            CREATE INDEX IF NOT EXISTS idx_probe_moad_test_pred_probe 
            ON probe_moad_test(probe_name);
        """)
        conn.commit()
        print("Successfully created probe_moad_test table")

def get_unique_proteins():
    """Get unique proteins from moad_test_results joined with processed_datasets."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT mtr.id, pd.pname, mtr.num_ligands
            FROM moad_test_results mtr
            JOIN processed_datasets pd ON mtr.id = pd.id
        """)
        return cur.fetchall()

def get_probes():
    """Get all probes from probe_set table."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT lname 
            FROM probe_set 
            ORDER BY id
        """)
        return [row[0] for row in cur.fetchall()]

def get_protein_pdb(protein_name):
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.pdb
            FROM proteins p
            WHERE p.name = %s
        """, (protein_name,))
        return cur.fetchone()[0].tobytes().decode('utf-8')

def get_ligand_sdf(lname):
    """Get ligand SDF content."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT mol 
            FROM ligands 
            WHERE name = %s
        """, (lname,))
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No SDF found for ligand {lname}")
        return result[0].tobytes().decode('utf-8')

def probe_proteins(proteins, probes, device='cuda', batch_size=8):
    """Run pocket prediction for all protein-probe pairs."""
    print(f"Using device: {device}")
    
    # Load model
    model_path = '../../models/moad_bs8_date06-06_time11-19-53.016800/last.ckpt'
    print("Loading model...")
    model = YuelPocket.load_from_checkpoint(model_path, map_location=device).eval().to(device)
    
    print(f"Loaded {len(probes)} probes")
    print(f"Loaded {len(proteins)} proteins")

    data_tuples = [(pname, probe) for _, pname, _ in proteins for probe in probes]
    print(f"Created dataset with {len(data_tuples)} protein-probe pairs")
    
    # Create dataset and dataloader
    dataset = PdbSdfDataset(data_pairs=data_tuples, pdb_cb=get_protein_pdb, sdf_cb=get_ligand_sdf, distance_cutoff=10.0, device='cpu')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate,
        num_workers=16,  # Number of worker processes
    )
    
    # Process batches
    idx = 0
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Move batch to GPU
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            pocket_pred = model(batch).squeeze(-1)  # shape: (batch_size, n_residues)
            protein_masks = batch['protein_mask'].squeeze(-1)  # shape: (batch_size, n_residues)
            for j in range(pocket_pred.shape[0]):
                if idx >= len(data_tuples):
                    break
                    
                protein_length = int(protein_masks[j].sum().item())
                pred = pocket_pred[j, :protein_length].cpu().numpy()
                preds.append((*data_tuples[idx], pred))
                idx += 1
    
    # Store results
    with db_connection() as conn:
        cur = conn.cursor()
        cur.executemany("""
            INSERT INTO probe_moad_test (protein_name, probe_name, pocket_pred)
            VALUES (%s, %s, %s)
            ON CONFLICT (protein_name, probe_name) DO UPDATE 
            SET pocket_pred = EXCLUDED.pocket_pred
        """, [(protein_name, probe_name, pickle.dumps(pred)) 
              for protein_name, probe_name, pred in preds])
        conn.commit()

if __name__ == "__main__":
    create_results_table()
    proteins = get_unique_proteins()
    probes = get_probes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_proteins(proteins, probes, device)

