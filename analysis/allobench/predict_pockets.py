#%%

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(__file__, '../../..'))
from src.db_utils import db_connection
import pickle
import matplotlib.pyplot as plt
from src.lightning import YuelPocket
import random
from src.datasets import collate
from yuel_pocket import PdbSdfDataset, predict_pocket
from cluster_allosteric import calculate_allosteric_clusters, plot_allosteric_clusters

def create_allobench_predictions_table():
    """Create a table to store pocket predictions for probe ligands on allobench proteins."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS allobench_predictions (
                pdb_id VARCHAR(4) NOT NULL,
                ligand_name TEXT NOT NULL,
                pocket_pred BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (pdb_id, ligand_name)
            );
            CREATE INDEX IF NOT EXISTS idx_allobench_pred_pdb ON allobench_predictions(pdb_id);
            CREATE INDEX IF NOT EXISTS idx_allobench_pred_ligand ON allobench_predictions(ligand_name);
        """)
        conn.commit()

def get_top_probes():
    """Get all probes from probe set."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT lname 
            FROM probe_set 
            ORDER BY id
        """)
        return [row[0] for row in cur.fetchall()]

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

def get_allobench_proteins():
    """Get proteins from allobench table where modulator_class is 'Lig'."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT protein_asd_id, pdb_id, pdb_content, 
                   in_allosteric_site, in_active_site
            FROM allobench
            WHERE in_allosteric_site IS NOT NULL 
            AND in_active_site IS NOT NULL
            AND modulator_class = 'Lig'
        """)
        return cur.fetchall()

def predict_pockets(probes, device, batch_size=16):
    print(f"Using device: {device}")
    
    # Load model
    model_path = '../../models/moad_bs8_date06-06_time11-19-53.016800/last.ckpt'
    print("Loading model...")
    model = YuelPocket.load_from_checkpoint(model_path, map_location=device).eval().to(device)
    
    # Get probe ligands
    all_probes = get_top_probes()
    probe_ligands = [all_probes[i] for i in probes]
    probe_sdfs = {ligand_name: get_ligand_sdf(ligand_name) for ligand_name in probe_ligands}
    print(f"Using probe ligands: {probe_ligands}")
    
    # Get all proteins with unique pdb_id
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT ON (pdb_id) pdb_id, pdb_content
            FROM allobench
            WHERE in_allosteric_site IS NOT NULL 
            AND in_active_site IS NOT NULL
            AND modulator_class = 'Lig'
            AND array_length(in_allosteric_site, 1) < 2000
        """)
        proteins = cur.fetchall()
    print(f"Found {len(proteins)} unique proteins in allobench with size < 2000 residues")
    
    # Prepare dataset inputs
    data_tuples = [[pdb_id, pdb_content, ligand_name, probe_sdfs[ligand_name]]
                   for pdb_id, pdb_content in proteins 
                   for ligand_name in probe_ligands]
    
    pdb_ids = [pdb_id for pdb_id, _, _, _ in data_tuples]
    ligand_names = [ligand_name for _, _, ligand_name, _ in data_tuples]
    data_pairs = [(pdb_content, sdf_content) for _, pdb_content, _, sdf_content in data_tuples]
    
    # Create dataset and dataloader
    print("Creating dataset and dataloader")
    dataset = PdbSdfDataset(data_pairs=data_pairs, distance_cutoff=10.0, device='cpu')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                           collate_fn=collate, num_workers=16,
                                           pin_memory=True)
    
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
                if idx >= len(pdb_ids):
                    break
                    
                protein_length = int(protein_masks[j].sum().item())
                pred = pocket_pred[j, :protein_length].cpu().numpy()
                preds.append((pdb_ids[idx], ligand_names[idx], pred))
                idx += 1
    
    # Store results
    with db_connection() as conn:
        cur = conn.cursor()
        cur.executemany("""
            INSERT INTO allobench_predictions (pdb_id, ligand_name, pocket_pred)
            VALUES (%s, %s, %s)
            ON CONFLICT (pdb_id, ligand_name) DO UPDATE 
            SET pocket_pred = EXCLUDED.pocket_pred
        """, [(pdb_id, ligand_name, pickle.dumps(pred)) for pdb_id, ligand_name, pred in preds])
        conn.commit()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda")
    # device = torch.device("mps")
    
    # Create predictions table
    # create_allobench_predictions_table()

    # First predict pockets
    # predict_pockets(probes=[11,12,13,14], device=device)
    

# %%
