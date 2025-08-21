#%%

import os
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import yuel_pocket
sys.path.append('..')
from src.db_utils import db_connection
from yuel_pocket import YuelPocket
from src.datasets import PocketDataset, get_dataloader

def evaluate_pocket_performance(pocket_pred, is_pocket, threshold=0.5):
    """Evaluate the performance of pocket prediction.
    
    Args:
        pocket_pred: shape (n_residues,) containing pocket predictions
        is_pocket: shape (n_residues,) containing true pocket labels
        threshold: threshold for converting predictions to binary labels
    
    Returns:
        dict containing evaluation metrics:
            accuracy: overall accuracy
            precision: precision score
            recall: recall score
            f1: F1 score
            mcc: Matthews Correlation Coefficient
    """
    # Convert predictions to binary using threshold
    pocket_pred_binary = (pocket_pred >= threshold).astype(int)
    
    # Calculate true positives, false positives, true negatives, false negatives
    tp = np.sum((pocket_pred_binary == 1) & (is_pocket == 1))
    fp = np.sum((pocket_pred_binary == 1) & (is_pocket == 0))
    tn = np.sum((pocket_pred_binary == 0) & (is_pocket == 0))
    fn = np.sum((pocket_pred_binary == 0) & (is_pocket == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate Matthews Correlation Coefficient
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator != 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc
    }

def create_results_table():
    """Create a new table to store prediction results."""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id SERIAL PRIMARY KEY,
                pname TEXT NOT NULL,
                lname TEXT NOT NULL,
                pocket_pred BYTEA NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pname, lname)
            )
        """)
        conn.commit()

def add_evaluation_columns():
    """Add evaluation metric columns to the existing test_results table."""
    with db_connection() as conn:
        cursor = conn.cursor()
        # Check if columns already exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'test_results'
        """)
        existing_columns = {row[0] for row in cursor.fetchall()}
        
        # Add new columns if they don't exist
        new_columns = {
            'accuracy': 'FLOAT',
            'precision': 'FLOAT',
            'recall': 'FLOAT',
            'f1': 'FLOAT',
            'mcc': 'FLOAT'
        }
        
        for column_name, column_type in new_columns.items():
            if column_name not in existing_columns:
                cursor.execute(f"""
                    ALTER TABLE test_results 
                    ADD COLUMN {column_name} {column_type}
                """)
                print(f"Added column {column_name} to test_results table")
        
        conn.commit()

def predict_and_save():
    """Predict pockets for test set and save results."""
    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '../models/moad_bs8_date27-05_time09-32-59.692193/last.ckpt'
    model = YuelPocket.load_from_checkpoint(model_path, map_location=device).eval().to(device)

    # Create results table
    create_results_table()

    # Create dataset and dataloader
    dataset = PocketDataset(device=device, split='test')
    dataloader = get_dataloader(dataset, batch_size=1, shuffle=False)
    print(f"Found {len(dataset)} test cases")

    # Process each test case with progress bar
    for i, batch in enumerate(tqdm(dataloader, desc="Predicting pockets")):
        try:
            pname = batch['pname'][0]
            lname = batch['lname'][0]

            # Print shapes for debugging
            # Check edge_index values
            n_nodes = batch['one_hot'].shape[1]
            edge_index = batch['edge_index'].squeeze(0)  # Remove batch dimension
            max_index = edge_index.max().item()
            min_index = edge_index.min().item()
            
            if max_index >= n_nodes or min_index < 0:
                tqdm.write(f"\nProcessing {pname} with {lname}")
                tqdm.write(f"one_hot shape: {batch['one_hot'].shape}")
                tqdm.write(f"edge_index shape: {batch['edge_index'].shape}")
                tqdm.write(f"edge_attr shape: {batch['edge_attr'].shape}")
                tqdm.write(f"node_mask shape: {batch['node_mask'].shape}")
                tqdm.write(f"edge_mask shape: {batch['edge_mask'].shape}")
                tqdm.write(f"protein_mask shape: {batch['protein_mask'].shape}")

                tqdm.write(f"Number of nodes: {n_nodes}")
                tqdm.write(f"Edge index range: [{min_index}, {max_index}]")
                tqdm.write(f"WARNING: Invalid edge indices found!")
                tqdm.write(f"Edge indices should be in range [0, {n_nodes-1}]")
                # Print some problematic edges
                invalid_edges = torch.where((edge_index >= n_nodes) | (edge_index < 0))
                tqdm.write(f"Found {len(invalid_edges[0])} invalid edges")
                if len(invalid_edges[0]) > 0:
                    tqdm.write("Sample of invalid edges:")
                    for i in range(min(5, len(invalid_edges[0]))):
                        edge_idx = invalid_edges[0][i]
                        tqdm.write(f"Edge {edge_idx}: {edge_index[edge_idx].tolist()}")
                continue

            # Get prediction
            with torch.no_grad():
                try:
                    pocket_pred = model.forward(batch)
                    # Remove all size-1 dimensions
                    pocket_pred = pocket_pred.squeeze()  # shape: (n_nodes,)
                    
                    # Get protein mask and remove all size-1 dimensions
                    protein_mask = batch['protein_mask'].squeeze()  # shape: (n_nodes,)
                    
                    # Apply protein mask to predictions
                    pocket_pred = pocket_pred * protein_mask
                    
                    # Convert to numpy
                    pocket_pred = pocket_pred.cpu().numpy()

                    # Save prediction to database using pickle
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO test_results (pname, lname, pocket_pred)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (pname, lname) 
                            DO UPDATE SET 
                                pocket_pred = EXCLUDED.pocket_pred,
                                created_at = CURRENT_TIMESTAMP
                        """, (pname, lname, pickle.dumps(pocket_pred)))
                        conn.commit()

                except RuntimeError as e:
                    if "CUDA error: device-side assert triggered" in str(e):
                        tqdm.write(f"CUDA error for {pname} with {lname}:")
                        tqdm.write(f"Error details: {str(e)}")
                        # Try to get more information about the tensors
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                tqdm.write(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                        continue
                    else:
                        raise e

        except Exception as e:
            tqdm.write(f"Error processing {pname} with {lname}: {str(e)}")
            continue

def evaluate_test_set():
    """Evaluate the performance of predictions on the test set."""
    # Add evaluation columns if they don't exist
    add_evaluation_columns()
    
    # Initialize metrics for averaging
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mcc': []
    }
    
    # Create dataset and dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PocketDataset(device=device, split='test')
    dataloader = get_dataloader(dataset, batch_size=1, shuffle=False)
    print(f"Found {len(dataset)} test cases")
    
    # Process each test case with progress bar
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating predictions")):
        try:
            pname = batch['pname'][0]
            lname = batch['lname'][0]
            
            # Get prediction from test_results and is_pocket from processed_datasets
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT tr.pocket_pred, pd.is_pocket
                    FROM test_results tr
                    JOIN processed_datasets pd ON tr.pname = pd.pname AND tr.lname = pd.lname
                    WHERE tr.pname = %s AND tr.lname = %s
                """, (pname, lname))
                result = cursor.fetchone()
                
                if result is None:
                    tqdm.write(f"No prediction found for {pname} with {lname}")
                    continue
                
                pocket_pred = pickle.loads(result[0])
                is_pocket = pickle.loads(result[1])
            
            # Evaluate performance
            metrics = evaluate_pocket_performance(pocket_pred, is_pocket)
            
            # Add metrics to running average
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
            
            # Update metrics in database
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE test_results 
                    SET accuracy = %s,
                        precision = %s,
                        recall = %s,
                        f1 = %s,
                        mcc = %s
                    WHERE pname = %s AND lname = %s
                """, (
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1'],
                    metrics['mcc'],
                    pname,
                    lname
                ))
                conn.commit()
            
        except Exception as e:
            tqdm.write(f"Error evaluating {pname} with {lname}: {str(e)}")
            continue
    
    # Print average metrics
    print("\nAverage performance metrics across all test cases:")
    for metric_name, values in all_metrics.items():
        if values:  # Check if we have any values
            avg_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric_name}: {avg_value:.4f} ± {std_value:.4f}")

if __name__ == '__main__':
    # Set environment variable for better CUDA error reporting
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    predict_and_save()
    evaluate_test_set()

# %%
