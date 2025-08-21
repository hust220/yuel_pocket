#%%

import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../../')
from src.db_utils import db_connection, add_column
import matplotlib.pyplot as plt
import pickle

def get_ligand_positions(lname):
    """Get ligand atom positions from database.
    
    Args:
        lname: ligand name to get ligand coordinates from database
    
    Returns:
        numpy array: ligand atom positions with shape (n_atoms, 3)
    """
    import pickle
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT molecule_pos 
            FROM raw_datasets 
            WHERE ligand_name = %s
        """, (lname,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No data found for ligand {lname}")
        
        # Convert bytea to numpy array using pickle
        ligand_pos = pickle.loads(result[0])
    
    return ligand_pos

def calculate_dcc_metrics():
    """Calculate DCC (Distance between Cluster Center and Ligand) metrics for all predictions.
    For each prediction:
    1. Get pocket_pred, protein positions, and ligand positions
    2. Calculate clusters
    3. For each cluster:
       - Get all atoms in the residues belonging to this cluster
       - Calculate minimum distance between any pocket atom and any ligand atom
       - Get maximum probability in cluster
    4. Store results back to database
    """
    # Add necessary columns if they don't exist
    add_column('moad_test_results', 'cluster_probs', 'BYTEA')
    add_column('moad_test_results', 'cluster_distances', 'BYTEA')
    
    with db_connection() as conn:
        cur = conn.cursor()
        # Get all rows from moad_test_results that have clusters and allatom_pos
        cur.execute("""
            SELECT m.id, p.lname, m.pocket_pred, m.clusters, m.protein_allatom_pos
            FROM moad_test_results m
            JOIN processed_datasets p ON m.id = p.id
            WHERE m.clusters IS NOT NULL AND m.protein_allatom_pos IS NOT NULL
        """)
        rows = cur.fetchall()
        
        print(f"Processing {len(rows)} rows")
        iprocessed = 0
        
        # Process each row
        for id, lname, pocket_pred, clusters, allatom_pos in tqdm(rows, total=len(rows), desc="Calculating DCC metrics"):
            try:
                # Unpickle data
                pocket_pred = pickle.loads(pocket_pred).squeeze()
                clusters = pickle.loads(clusters)
                allatom_pos = pickle.loads(allatom_pos)  # This is a list of lists, each inner list contains atom positions for one residue
                
                # Get ligand positions
                ligand_pos = get_ligand_positions(lname)
                
                # Ensure all arrays have matching lengths
                min_length = min(len(pocket_pred), len(clusters), len(allatom_pos))
                if len(pocket_pred) != min_length or len(clusters) != min_length:
                    pocket_pred = pocket_pred[:min_length]
                    clusters = clusters[:min_length]
                
                # Get unique clusters (excluding noise)
                unique_clusters = np.unique(clusters)
                unique_clusters = unique_clusters[unique_clusters != -1]
                
                # Initialize dictionaries for cluster metrics
                cluster_probs = {}  # Maximum probability for each cluster
                cluster_distances = {}  # Minimum distance to ligand for each cluster
                
                # Calculate metrics for each cluster
                for cluster_id in unique_clusters:
                    # Get residue indices for this cluster
                    cluster_mask = (clusters == cluster_id)
                    cluster_residue_indices = np.where(cluster_mask)[0]
                    
                    # Get all atom positions for these residues
                    cluster_atom_positions = []
                    for residue_idx in cluster_residue_indices:
                        cluster_atom_positions.extend(allatom_pos[residue_idx])
                    cluster_atom_positions = np.array(cluster_atom_positions)
                    
                    # Calculate minimum distance between any pocket atom and any ligand atom
                    # Reshape arrays for broadcasting
                    pocket_atoms = cluster_atom_positions.reshape(-1, 1, 3)  # Shape: (n_pocket_atoms, 1, 3)
                    ligand_atoms = ligand_pos.reshape(1, -1, 3)  # Shape: (1, n_ligand_atoms, 3)
                    
                    # Calculate all pairwise distances
                    distances = np.sqrt(np.sum((pocket_atoms - ligand_atoms) ** 2, axis=2))  # Shape: (n_pocket_atoms, n_ligand_atoms)
                    
                    # Get minimum distance
                    min_distance = float(np.min(distances))  # Convert to native Python type for better serialization
                    cluster_distances[int(cluster_id)] = min_distance
                    
                    # Get maximum probability for this cluster
                    max_prob = float(np.max(pocket_pred[cluster_mask]))  # Convert to native Python type
                    cluster_probs[int(cluster_id)] = max_prob
                
                # Store results back to database
                cur.execute("""
                    UPDATE moad_test_results 
                    SET cluster_probs = %s,
                        cluster_distances = %s
                    WHERE id = %s
                """, (pickle.dumps(cluster_probs), pickle.dumps(cluster_distances), id))
                conn.commit()
                iprocessed += 1
                
            except Exception as e:
                print(f"Error processing {lname} (id: {id}): {str(e)}")
                print(f"pocket_pred shape: {pocket_pred.shape if 'pocket_pred' in locals() else 'not loaded'}")
                print(f"clusters shape: {clusters.shape if 'clusters' in locals() else 'not loaded'}")
                print(f"allatom_pos length: {len(allatom_pos) if 'allatom_pos' in locals() else 'not loaded'}")
                print(f"ligand_pos shape: {ligand_pos.shape if 'ligand_pos' in locals() else 'not loaded'}")
                raise e
        
        print(f"Successfully processed {iprocessed} rows")

def plot_dcc_success_rate(save_dir='plots'):
    """Plot success rate curves based on DCC thresholds.
    Four curves:
    1. Success rate where the cluster with DCC < threshold has top 1 probability
    2. Success rate where the cluster with DCC < threshold has probability ranked in top 3
    3. Success rate where the cluster with DCC < threshold has probability ranked in top 5
    4. Success rate where the cluster with DCC < threshold has probability ranked in all ligands
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all results from database
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT cluster_probs, cluster_distances, num_ligands
            FROM moad_test_results
            WHERE cluster_probs IS NOT NULL AND cluster_distances IS NOT NULL
        """)
        results = cur.fetchall()
    
    # Prepare data for different DCC thresholds
    thresholds = np.arange(2, 20.1, 0.5)  # From 2Å to 20Å with 0.5Å steps
    success_rates_top1 = []  # For top 1 prob
    success_rates_top3 = []  # For top 3 prob
    success_rates_top5 = []  # For top 5 prob
    success_rates_all = []   # For all ligands
    
    print(f"Processing {len(results)} predictions...")
    
    for threshold in tqdm(thresholds, desc="Calculating success rates"):
        n_success_top1 = 0
        n_success_top3 = 0
        n_success_top5 = 0
        n_success_all = 0
        n_valid = 0
        
        for cluster_probs, cluster_distances, num_ligands in results:
            try:
                # Unpickle data
                probs = pickle.loads(cluster_probs)
                distances = pickle.loads(cluster_distances)
                
                if not probs or not distances:  # Skip if empty
                    continue
                
                # Convert to lists for sorting
                prob_items = list(probs.items())  # [(cluster_id, prob), ...]
                
                # Find clusters with DCC < threshold
                close_clusters = [cid for cid, dist in distances.items() if dist < threshold]
                
                # Count this as a valid prediction
                n_valid += 1
                
                if not close_clusters:  # No clusters are close enough - count as failure
                    continue
                
                # Get probabilities of close clusters
                close_probs = [probs[cid] for cid in close_clusters]
                max_close_prob = max(close_probs)
                
                # Sort all probabilities in descending order
                sorted_probs = sorted(probs.values(), reverse=True)
                                
                # Check if any close cluster's probability is in top N
                max_close_rank = min(sorted_probs.index(prob) + 1 for prob in close_probs)
                if max_close_rank <= num_ligands:
                    n_success_top1 += 1
                if max_close_rank <= 2 + num_ligands:
                    n_success_top3 += 1
                if max_close_rank <= 4 + num_ligands:
                    n_success_top5 += 1
                n_success_all += 1  # Always count for all ligands case
                
            except Exception as e:
                print(f"Error processing a prediction: {str(e)}")
                continue
        
        # Calculate success rates
        if n_valid > 0:
            success_rates_top1.append(n_success_top1 / n_valid)
            success_rates_top3.append(n_success_top3 / n_valid)
            success_rates_top5.append(n_success_top5 / n_valid)
            success_rates_all.append(n_success_all / n_valid)
        else:
            success_rates_top1.append(0)
            success_rates_top3.append(0)
            success_rates_top5.append(0)
            success_rates_all.append(0)
    
    # Plot curves
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(thresholds, success_rates_top1, '-', color='#43a3ef', label='N')
    plt.plot(thresholds, success_rates_top3, '-', color='#ef767b', label='N+2')
    # plt.plot(thresholds, success_rates_top5, '--', color='#43a3ef', label='Top 5')
    plt.plot(thresholds, success_rates_all, '-', color='black', label='All')
    plt.xlabel('Distance Threshold (Å)')
    plt.ylabel('Success Rate')
    plt.grid(True, alpha=0.3)
    
    # Configure legend without frame
    legend = plt.legend(frameon=False)
    
    # Save plot
    save_path = os.path.join(save_dir, 'dcc_success_rate.svg')
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    # Print some statistics
    print("\nSuccess Rate Statistics:")
    print(f"Number of valid predictions: {n_valid}")
    print("\nTop 1 Success Rate:")
    print(f"Best: {max(success_rates_top1):.3f} at threshold {thresholds[np.argmax(success_rates_top1)]:.1f}Å")
    print(f"Worst: {min(success_rates_top1):.3f} at threshold {thresholds[np.argmin(success_rates_top1)]:.1f}Å")
    print("\nTop 3 Success Rate:")
    print(f"Best: {max(success_rates_top3):.3f} at threshold {thresholds[np.argmax(success_rates_top3)]:.1f}Å")
    print(f"Worst: {min(success_rates_top3):.3f} at threshold {thresholds[np.argmin(success_rates_top3)]:.1f}Å")
    print("\nTop 5 Success Rate:")
    print(f"Best: {max(success_rates_top5):.3f} at threshold {thresholds[np.argmax(success_rates_top5)]:.1f}Å")
    print(f"Worst: {min(success_rates_top5):.3f} at threshold {thresholds[np.argmin(success_rates_top5)]:.1f}Å")
    print("\nAll Success Rate:")
    print(f"Best: {max(success_rates_all):.3f} at threshold {thresholds[np.argmax(success_rates_all)]:.1f}Å")
    print(f"Worst: {min(success_rates_all):.3f} at threshold {thresholds[np.argmin(success_rates_all)]:.1f}Å")

if __name__ == "__main__":    

    # calculate_dcc_metrics()
    
    plot_dcc_success_rate()
    

# %%
