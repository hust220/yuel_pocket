import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
import sys
sys.path.append('../../')
from src.lightning import YuelPocket
from src.datasets import PocketDataset, get_dataloader, collate
from src.db_utils import db_connection

def create_results_table():
    """Create results table if not exists"""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moad_test_results (
                id INTEGER PRIMARY KEY,
                pocket_pred BYTEA
            )
        """)
        conn.commit()
        cursor.close()

def run_test(model_path, device='cuda'):
    """Test model performance on test set."""
    # Load model
    model = YuelPocket.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()
    
    # Get dataset info and filter out already processed proteins
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id from moad_test_results
        """)
        processed_ids = {row[0] for row in cursor.fetchall()}
        cursor.execute("""
            SELECT id, lname, in_coach420, in_holo4k from processed_datasets
            WHERE split = 'test'
        """)
        dataset_info = {row[0]: {'lname': row[1], 'in_coach420': row[2], 'in_holo4k': row[3]} for row in cursor.fetchall()}
        all_ids = set(dataset_info.keys())
        lname_to_id = {v['lname']: k for k,v in dataset_info.items()}
        cursor.close()

        remaining_ids = all_ids - processed_ids
        
    print(f"Processing {len(remaining_ids)} remaining proteins...")
    
    # Get test dataset and dataloader
    test_dataset = PocketDataset(device=device, split='test')
    test_loader = get_dataloader(test_dataset, batch_size=1, collate_fn=collate)
    
    # Process each batch
    iprocessed = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing test set"):
            try:
                # Move batch to device
                device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                lname = device_batch['lname'][0]
                id = lname_to_id[lname]
                if id not in remaining_ids:
                    continue
                
                pred = model(device_batch)
                pocket_pred = pred.cpu().numpy()
                
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO moad_test_results (id, pocket_pred)
                        VALUES (%s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET pocket_pred = EXCLUDED.pocket_pred
                    """, (id, pickle.dumps(pocket_pred)))
                    conn.commit()
                    cursor.close()
                iprocessed += 1
            except Exception as e:
                print(f"Error processing {lname}: {str(e)}")
                # raise e
    
    print(f"Test results have been saved to moad_test_results table. Processed {iprocessed} proteins.")

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model path
    model_path = '../../yuel_pocket.ckpt'
    
    # Create results table if it doesn't exist
    create_results_table()
    
    # Run evaluation
    run_test(model_path, device) 