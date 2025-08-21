#%%

import torch
import sys
sys.path.append('../../')
from src.datasets import PocketDataset
from src.db_utils import db_connection

def get_unique_proteins():
    with db_connection('yuel_pocket') as conn:
        cur = conn.cursor()
        
        # Get unique proteins from processed_datasets
        cur.execute("SELECT DISTINCT pname FROM processed_datasets;")
        processed_proteins = {row[0] for row in cur.fetchall()}
        
        # Get proteins from COACH420 and remove chain identifiers
        cur.execute("SELECT name FROM coach420_proteins;")
        coach420_proteins = {row[0][:-1] for row in cur.fetchall()}
        
        # Get proteins from Holo4K
        cur.execute("SELECT name FROM holo4k_proteins;")
        holo4k_proteins = {row[0] for row in cur.fetchall()}
        
        return processed_proteins, coach420_proteins, holo4k_proteins

def analyze_overlap():
    processed_proteins, coach420_proteins, holo4k_proteins = get_unique_proteins()
    
    # Calculate overlaps
    coach420_overlap = processed_proteins.intersection(coach420_proteins)
    holo4k_overlap = processed_proteins.intersection(holo4k_proteins)
    both_overlap = coach420_overlap.intersection(holo4k_overlap)
    
    # Print results
    print(f"Total unique proteins in processed datasets: {len(processed_proteins)}")
    print(f"Total proteins in COACH420 (after removing chain IDs): {len(coach420_proteins)}")
    print(f"Total proteins in Holo4K: {len(holo4k_proteins)}")
    print("\nOverlap Analysis:")
    print(f"Proteins in processed datasets that are in COACH420: {len(coach420_overlap)}")
    print(f"Proteins in processed datasets that are in Holo4K: {len(holo4k_overlap)}")
    print(f"Proteins in processed datasets that are in both COACH420 and Holo4K: {len(both_overlap)}")
    
    # Print some example overlapping proteins
    if coach420_overlap:
        print("\nExample proteins in processed datasets and COACH420:")
        print(list(coach420_overlap)[:5])
    
    if holo4k_overlap:
        print("\nExample proteins in processed datasets and Holo4K:")
        print(list(holo4k_overlap)[:5])


def check_protein_data(protein_name, device='cpu'):
    """Check data for a specific protein."""
    # Get test dataset
    test_dataset = PocketDataset(device=device, split='test')
    
    # Find the protein in the dataset
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, in_coach420, in_holo4k
            FROM processed_datasets
            WHERE split = 'test' AND pname = %s
        """, (protein_name,))
        result = cursor.fetchone()
        if not result:
            print(f"Protein {protein_name} not found in test set")
            return
        protein_id, is_coach420, is_holo4k = result
        cursor.close()
    
    print(f"\nChecking protein: {protein_name}")
    print(f"Dataset info: COACH420={is_coach420}, Holo4K={is_holo4k}")
    
    # Find the protein's index in the dataset
    found = False
    for i in range(len(test_dataset)):
        try:
            data = test_dataset[i]
            if data['pname'] == protein_name:
                found = True
                break
        except Exception as e:
            print(f"Error accessing index {i}: {str(e)}")
            continue
    
    if not found:
        print(f"Could not find protein {protein_name} in dataset")
        return
    
    print(f"Found protein at index {i}")
    
    # Print shapes and basic info
    print("\nData shapes:")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            if k in ['one_hot', 'edge_index', 'edge_attr']:
                print(f"  - min: {v.min().item():.3f}, max: {v.max().item():.3f}")
                print(f"  - mean: {v.float().mean().item():.3f}")
                if torch.isnan(v).any():
                    print(f"  - Contains NaN values")
                if torch.isinf(v).any():
                    print(f"  - Contains Inf values")
        else:
            print(f"{k}: type={type(v)}")
    
    # Check edge indices
    if 'edge_index' in data:
        edge_index = data['edge_index']
        max_node_idx = data['one_hot'].shape[0] - 1
        invalid_edges = torch.where((edge_index < 0) | (edge_index > max_node_idx))
        if len(invalid_edges[0]) > 0:
            print(f"\nFound {len(invalid_edges[0])} invalid edge indices")
            print(f"Max valid node index: {max_node_idx}")
            print(f"Invalid edge indices: {edge_index[invalid_edges]}")
    
    # Check masks
    if 'protein_mask' in data and 'is_pocket' in data:
        protein_mask = data['protein_mask']
        is_pocket = data['is_pocket']
        print(f"\nMask statistics:")
        print(f"Protein mask sum: {protein_mask.sum().item()}")
        print(f"Pocket mask sum: {is_pocket.sum().item()}")
        print(f"Overlap: {(protein_mask * is_pocket).sum().item()}")

if __name__ == "__main__":
    # Example usage
    protein_name = "1a0q"  # Replace with actual protein name
    check_protein_data(protein_name) 

    analyze_overlap()

# %%

