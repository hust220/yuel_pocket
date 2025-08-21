# %%

import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
sys.path.append('../..')
from src.db_utils import db_connection, add_column
from src.lightning import YuelPocket
from src.pdb_utils import Structure
from yuel_pocket import _predict_pocket, parse_protein_structure
import io
from sklearn.cluster import DBSCAN
import time

def create_pdbbind_test_table():
    """Create a table to store pocket prediction results"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pdbbind_test (
                id SERIAL PRIMARY KEY,
                pdb_id TEXT UNIQUE NOT NULL,
                pocket_pred BYTEA
            )
        """)
        
        # Create indexes for faster JOIN operations
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pdbbind_pdb_id ON pdbbind(pdb_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pdbbind_test_pdb_id ON pdbbind_test(pdb_id)")
        
        conn.commit()

def ensure_indexes_exist():
    """Ensure all necessary indexes exist for optimal query performance."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create indexes if they don't exist
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pdbbind_pdb_id ON pdbbind(pdb_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pdbbind_test_pdb_id ON pdbbind_test(pdb_id)")
        
        # Also create indexes on binding_type for filtering
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pdbbind_binding_type ON pdbbind(binding_type)")
        
        conn.commit()
        print("Database indexes ensured for optimal performance.")

def predict_pocket_for_pdbbind():
    """Process each entry in pdbbind table and predict pockets"""
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '../../models/moad_bs8_date06-06_time11-19-53.016800/last.ckpt'
    model = YuelPocket.load_from_checkpoint(model_path, map_location=device).eval().to(device)
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get all PDBbind entries
        cur.execute("SELECT pdb_id, receptor_pdb, ligand_sdf FROM pdbbind")
        entries = cur.fetchall()
        
        # Process each entry
        for pdb_id, receptor_pdb, ligand_sdf in tqdm(entries, desc="Processing PDBbind entries"):
            try:
                # Convert binary data to string
                pdb_content = receptor_pdb.tobytes().decode('utf-8')
                sdf_content = ligand_sdf.tobytes().decode('utf-8')
                
                # Run prediction with string content
                pocket_pred, _ = _predict_pocket(
                    pdb_content,
                    sdf_content,
                    model,
                    distance_cutoff=10.0,
                    device=device
                )
                
                # Convert predictions to Python list for database storage
                pocket_pred = pocket_pred.cpu().numpy().squeeze()
                
                # Store results
                cur.execute("""
                    INSERT INTO pdbbind_test (pdb_id, pocket_pred)
                    VALUES (%s, %s)
                    ON CONFLICT (pdb_id) DO UPDATE 
                    SET pocket_pred = EXCLUDED.pocket_pred
                """, (pdb_id, pickle.dumps(pocket_pred)))
                conn.commit()
                
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                raise e
                continue

def convert_to_molar(value, unit):
    """Convert a value from its unit to Molar concentration."""
    unit_to_multiplier = {
        'mM': 1e-3,   # millimolar to molar
        'uM': 1e-6,   # micromolar to molar
        'nM': 1e-9,   # nanomolar to molar
        'pM': 1e-12,  # picomolar to molar
        'fM': 1e-15   # femtomolar to molar
    }
    return value * unit_to_multiplier[unit]

def load_binding_data():
    """Load binding data from database."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.pdb_id, p.binding_type, p.binding_value, p.binding_unit, 
                   t.pocket_pred, t.clusters, t.protein_pos
            FROM pdbbind p
            JOIN pdbbind_test t ON p.pdb_id = t.pdb_id
            WHERE p.binding_type IS NOT NULL
              AND p.binding_value IS NOT NULL
              AND p.binding_unit IS NOT NULL
              AND t.pocket_pred IS NOT NULL
              AND t.clusters IS NOT NULL
              AND p.binding_type IN ('Kd', 'Ki')  -- Only include Kd and Ki
        """)
        return cur.fetchall()

def get_all_pdb_ids():
    """Get all PDB IDs that have valid binding and prediction data."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.pdb_id
            FROM pdbbind p
            INNER JOIN pdbbind_test t ON p.pdb_id = t.pdb_id
            WHERE p.binding_type IS NOT NULL
              AND p.binding_value IS NOT NULL
              AND p.binding_unit IS NOT NULL
              AND t.pocket_pred IS NOT NULL
              AND t.clusters IS NOT NULL
              AND p.binding_type IN ('Kd', 'Ki')  -- Only include Kd and Ki
            ORDER BY p.pdb_id
        """)
        return [row[0] for row in cur.fetchall()]

def analyze_max_probability(pdb_ids, batch_size=100):
    """Analyze correlation between maximum probability and binding affinity."""
    pdb_ids_acc = []
    binding_types_acc = []
    pk_values_acc = []
    max_preds_acc = []
    
    # Process data in batches
    for i in tqdm(range(0, len(pdb_ids), batch_size), desc="Analyzing max probability"):
        batch_ids = pdb_ids[i:i + batch_size]
        
        # Get data for this batch
        with db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT p.pdb_id, p.binding_type, p.binding_value, p.binding_unit, t.pocket_pred
                FROM pdbbind p
                JOIN pdbbind_test t ON p.pdb_id = t.pdb_id
                WHERE p.pdb_id = ANY(%s)
            """, (batch_ids,))
            batch_data = cur.fetchall()
        
        # Process each entry in the batch
        for pdb_id, binding_type, binding_value, binding_unit, pocket_pred in batch_data:
            try:
                # Convert binding value to pK
                molar_value = convert_to_molar(binding_value, binding_unit)
                pk_value = -np.log10(molar_value)
                
                # Get max prediction
                pred_array = pickle.loads(pocket_pred).squeeze()
                max_pred = np.max(pred_array)
                
                pdb_ids_acc.append(pdb_id)
                binding_types_acc.append(binding_type)
                pk_values_acc.append(pk_value)
                max_preds_acc.append(max_pred)
                
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
    
    # Calculate correlations
    correlation, p_value = stats.pearsonr(pk_values_acc, max_preds_acc)
    spearman_corr, spearman_p = stats.spearmanr(pk_values_acc, max_preds_acc)
    
    return {
        'pdb_ids': pdb_ids_acc,
        'binding_types': binding_types_acc,
        'pk_values': np.array(pk_values_acc),
        'predictions': np.array(max_preds_acc),
        'pearson': (correlation, p_value),
        'spearman': (spearman_corr, spearman_p)
    }

def analyze_mean_probability(pdb_ids, batch_size=100):
    """Analyze correlation between mean probability (of probabilities > 0.05) and binding affinity."""
    pdb_ids_acc = []
    binding_types_acc = []
    pk_values_acc = []
    mean_preds_acc = []
    
    # Process data in batches
    for i in tqdm(range(0, len(pdb_ids), batch_size), desc="Analyzing mean probability"):
        batch_ids = pdb_ids[i:i + batch_size]
        
        # Get data for this batch
        t1 = time.time()
        with db_connection() as conn:
            cur = conn.cursor()
            # Use IN clause instead of ANY for better performance
            placeholders = ','.join(['%s'] * len(batch_ids))
            cur.execute(f"""
                SELECT p.pdb_id, p.binding_type, p.binding_value, p.binding_unit, t.pocket_pred
                FROM pdbbind p
                INNER JOIN pdbbind_test t ON p.pdb_id = t.pdb_id
                WHERE p.pdb_id IN ({placeholders})
            """, batch_ids)
            batch_data = cur.fetchall()
        t2 = time.time()
        print(f"Time taken to fetch data: {t2 - t1} seconds")

        # Process each entry in the batch
        for pdb_id, binding_type, binding_value, binding_unit, pocket_pred in batch_data:
            try:
                # Convert binding value to pK
                molar_value = convert_to_molar(binding_value, binding_unit)
                pk_value = -np.log10(molar_value)
                
                # Get mean prediction for probabilities > 0.05
                pred_array = pickle.loads(pocket_pred).squeeze()
                high_prob_mask = pred_array > 0.05
                if np.any(high_prob_mask):  # Only include if there are probabilities > 0.05
                    mean_pred = np.mean(pred_array[high_prob_mask])
                    
                    pdb_ids_acc.append(pdb_id)
                    binding_types_acc.append(binding_type)
                    pk_values_acc.append(pk_value)
                    mean_preds_acc.append(mean_pred)
                
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
        t3 = time.time()
        print(f"Time taken to process batch: {t3 - t2} seconds")
    
    # Calculate correlations
    correlation, p_value = stats.pearsonr(pk_values_acc, mean_preds_acc)
    spearman_corr, spearman_p = stats.spearmanr(pk_values_acc, mean_preds_acc)
    
    return {
        'pdb_ids': pdb_ids_acc,
        'binding_types': binding_types_acc,
        'pk_values': np.array(pk_values_acc),
        'predictions': np.array(mean_preds_acc),
        'pearson': (correlation, p_value),
        'spearman': (spearman_corr, spearman_p)
    }

def analyze_probability_sum(pdb_ids, batch_size=100, prob_threshold=0.05):
    """Analyze correlation between sum of probabilities (>0.05) and binding affinity."""
    pdb_ids_acc = []
    binding_types_acc = []
    pk_values_acc = []
    sum_preds_acc = []
    
    # Process data in batches
    for i in tqdm(range(0, len(pdb_ids), batch_size), desc="Analyzing probability sum"):
        batch_ids = pdb_ids[i:i + batch_size]
        
        # Get data for this batch
        with db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT p.pdb_id, p.binding_type, p.binding_value, p.binding_unit, t.pocket_pred
                FROM pdbbind p
                JOIN pdbbind_test t ON p.pdb_id = t.pdb_id
                WHERE p.pdb_id = ANY(%s)
            """, (batch_ids,))
            batch_data = cur.fetchall()
        
        # Process each entry in the batch
        for pdb_id, binding_type, binding_value, binding_unit, pocket_pred in batch_data:
            try:
                # Convert binding value to pK
                molar_value = convert_to_molar(binding_value, binding_unit)
                pk_value = -np.log10(molar_value)
                
                # Get sum of predictions above threshold
                pred_array = pickle.loads(pocket_pred).squeeze()
                # Only sum probabilities above threshold
                high_prob_mask = pred_array > prob_threshold
                sum_pred = np.sum(pred_array[high_prob_mask])
                
                pdb_ids_acc.append(pdb_id)
                binding_types_acc.append(binding_type)
                pk_values_acc.append(pk_value)
                sum_preds_acc.append(sum_pred)
                
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
    
    # Calculate correlations
    correlation, p_value = stats.pearsonr(pk_values_acc, sum_preds_acc)
    spearman_corr, spearman_p = stats.spearmanr(pk_values_acc, sum_preds_acc)
    
    return {
        'pdb_ids': pdb_ids_acc,
        'binding_types': binding_types_acc,
        'pk_values': np.array(pk_values_acc),
        'predictions': np.array(sum_preds_acc),
        'pearson': (correlation, p_value),
        'spearman': (spearman_corr, spearman_p)
    }

def analyze_top_cluster_mean(pdb_ids, batch_size=100):
    """Analyze correlation between mean probability of highest probability cluster and binding affinity."""
    pdb_ids_acc = []
    binding_types_acc = []
    pk_values_acc = []
    cluster_means_acc = []
    
    # Process data in batches
    for i in tqdm(range(0, len(pdb_ids), batch_size), desc="Analyzing cluster means"):
        batch_ids = pdb_ids[i:i + batch_size]
        
        # Get data for this batch
        with db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT p.pdb_id, p.binding_type, p.binding_value, p.binding_unit, 
                       t.pocket_pred, t.clusters
                FROM pdbbind p
                JOIN pdbbind_test t ON p.pdb_id = t.pdb_id
                WHERE p.pdb_id = ANY(%s)
            """, (batch_ids,))
            batch_data = cur.fetchall()
        
        # Process each entry in the batch
        for pdb_id, binding_type, binding_value, binding_unit, pocket_pred, clusters in batch_data:
            try:
                # Convert binding value to pK
                molar_value = convert_to_molar(binding_value, binding_unit)
                pk_value = -np.log10(molar_value)
                
                # Get predictions and clusters
                pred_array = pickle.loads(pocket_pred).squeeze()
                cluster_labels = pickle.loads(clusters)
                
                # Ensure arrays have the same length
                min_length = min(len(pred_array), len(cluster_labels))
                if len(pred_array) != min_length or len(cluster_labels) != min_length:
                    # print(f"Warning: Length mismatch for {pdb_id}. Truncating arrays to {min_length}")
                    pred_array = pred_array[:min_length]
                    cluster_labels = cluster_labels[:min_length]
                
                # Find the cluster with highest max probability
                unique_clusters = np.unique(cluster_labels)
                unique_clusters = unique_clusters[unique_clusters != -1]  # Remove noise cluster
                
                if len(unique_clusters) == 0:
                    continue
                
                max_prob_cluster = -1
                max_prob = -1
                for cluster_id in unique_clusters:
                    cluster_mask = (cluster_labels == cluster_id)
                    cluster_probs = pred_array[cluster_mask]
                    cluster_max_prob = np.max(cluster_probs)
                    if cluster_max_prob > max_prob:
                        max_prob = cluster_max_prob
                        max_prob_cluster = cluster_id
                
                # Calculate mean probability for the highest probability cluster
                top_cluster_mask = (cluster_labels == max_prob_cluster)
                top_cluster_mean = np.mean(pred_array[top_cluster_mask])
                
                pdb_ids_acc.append(pdb_id)
                binding_types_acc.append(binding_type)
                pk_values_acc.append(pk_value)
                cluster_means_acc.append(top_cluster_mean)
                
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
    
    # Calculate correlations
    correlation, p_value = stats.pearsonr(pk_values_acc, cluster_means_acc)
    spearman_corr, spearman_p = stats.spearmanr(pk_values_acc, cluster_means_acc)
    
    return {
        'pdb_ids': pdb_ids_acc,
        'binding_types': binding_types_acc,
        'pk_values': np.array(pk_values_acc),
        'predictions': np.array(cluster_means_acc),
        'pearson': (correlation, p_value),
        'spearman': (spearman_corr, spearman_p)
    }

def analyze_top_3_clusters_mean(pdb_ids, batch_size=100):
    """Analyze correlation between mean probability of top 3 highest probability clusters and binding affinity."""
    pdb_ids_acc = []
    binding_types_acc = []
    pk_values_acc = []
    cluster_means_acc = []
    
    # Process data in batches
    for i in tqdm(range(0, len(pdb_ids), batch_size), desc="Analyzing top 3 clusters mean"):
        batch_ids = pdb_ids[i:i + batch_size]
        
        # Get data for this batch
        with db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT p.pdb_id, p.binding_type, p.binding_value, p.binding_unit, 
                       t.pocket_pred, t.clusters
                FROM pdbbind p
                JOIN pdbbind_test t ON p.pdb_id = t.pdb_id
                WHERE p.pdb_id = ANY(%s)
            """, (batch_ids,))
            batch_data = cur.fetchall()
        
        # Process each entry in the batch
        for pdb_id, binding_type, binding_value, binding_unit, pocket_pred, clusters in batch_data:
            try:
                # Convert binding value to pK
                molar_value = convert_to_molar(binding_value, binding_unit)
                pk_value = -np.log10(molar_value)
                
                # Get predictions and clusters
                pred_array = pickle.loads(pocket_pred).squeeze()
                cluster_labels = pickle.loads(clusters)
                
                # Ensure arrays have the same length
                min_length = min(len(pred_array), len(cluster_labels))
                if len(pred_array) != min_length or len(cluster_labels) != min_length:
                    pred_array = pred_array[:min_length]
                    cluster_labels = cluster_labels[:min_length]
                
                # Find the top 3 clusters with highest max probability
                unique_clusters = np.unique(cluster_labels)
                unique_clusters = unique_clusters[unique_clusters != -1]  # Remove noise cluster
                
                if len(unique_clusters) == 0:
                    continue
                
                # Calculate max probability for each cluster
                cluster_max_probs = []
                for cluster_id in unique_clusters:
                    cluster_mask = (cluster_labels == cluster_id)
                    cluster_probs = pred_array[cluster_mask]
                    cluster_max_prob = np.max(cluster_probs)
                    cluster_max_probs.append((cluster_max_prob, cluster_id))
                
                # Sort clusters by max probability and get top 3
                cluster_max_probs.sort(reverse=True)
                top_3_clusters = cluster_max_probs[:3]
                
                # Calculate mean probability for the top 3 clusters
                if len(top_3_clusters) > 0:
                    top_3_mask = np.zeros_like(cluster_labels, dtype=bool)
                    for _, cluster_id in top_3_clusters:
                        top_3_mask |= (cluster_labels == cluster_id)
                    top_3_mean = np.mean(pred_array[top_3_mask])
                    
                    pdb_ids_acc.append(pdb_id)
                    binding_types_acc.append(binding_type)
                    pk_values_acc.append(pk_value)
                    cluster_means_acc.append(top_3_mean)
                
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
    
    # Calculate correlations
    correlation, p_value = stats.pearsonr(pk_values_acc, cluster_means_acc)
    spearman_corr, spearman_p = stats.spearmanr(pk_values_acc, cluster_means_acc)
    
    return {
        'pdb_ids': pdb_ids_acc,
        'binding_types': binding_types_acc,
        'pk_values': np.array(pk_values_acc),
        'predictions': np.array(cluster_means_acc),
        'pearson': (correlation, p_value),
        'spearman': (spearman_corr, spearman_p)
    }

def plot_correlation_analysis(metrics, analysis_name, save_dir='plots'):
    """Plot correlation analysis results."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(3.5, 2.5))
    
    # Create scatter plot
    colors = {'Kd': '#43a3ef', 'Ki': '#ef767b'}
    for btype in colors:
        mask = np.array(metrics['binding_types']) == btype
        if np.any(mask):
            plt.scatter(
                metrics['pk_values'][mask], 
                metrics['predictions'][mask],
                c=colors[btype], 
                label=btype, 
                alpha=0.5,
                s=10
            )
    
    pearson_r, pearson_p = metrics['pearson']
    spearman_r, spearman_p = metrics['spearman']
    
    plt.xlabel('Binding Affinity (-log(M))')
    plt.ylabel(f'{analysis_name.replace("_", " ").title()}')
    
    # Auto-calculate y-axis limits with space for legend
    y_min = np.min(metrics['predictions']) * 0.9  # 90% of min to leave some space below
    y_max = np.max(metrics['predictions']) * 1.1  # 115% of max to leave space for legend
    plt.ylim(y_min, y_max)
    
    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.05))
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_file = os.path.join(save_dir, f'{analysis_name}_correlation.svg')
    print(f"\nSaving plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def print_correlation_statistics(metrics, analysis_name):
    """Print detailed statistics for the analysis."""
    print(f"\n{analysis_name.replace('_', ' ').title()} Analysis:")
    print("-" * 50)
    print(f"Number of samples: {len(metrics['pk_values'])}")
    
    pearson_r, pearson_p = metrics['pearson']
    spearman_r, spearman_p = metrics['spearman']
    
    print(f"\nCorrelation Results:")
    print(f"Pearson correlation: {pearson_r:.3f} (p={pearson_p:.3e})")
    print(f"Spearman correlation: {spearman_r:.3f} (p={spearman_p:.3e})")
    
    # Calculate statistics by binding type
    for btype in ['Kd', 'Ki']:
        mask = np.array(metrics['binding_types']) == btype
        if np.sum(mask) > 2:
            r, p = stats.pearsonr(metrics['pk_values'][mask], metrics['predictions'][mask])
            rs, ps = stats.spearmanr(metrics['pk_values'][mask], metrics['predictions'][mask])
            print(f"\n{btype} (n={np.sum(mask)}):")
            print(f"  Pearson R: {r:.3f} (p={p:.3e})")
            print(f"  Spearman R: {rs:.3f} (p={ps:.3e})")

def analyze_binding_correlation():
    """Analyze correlations between binding affinity and pocket predictions."""
    # Ensure indexes exist for optimal performance
    ensure_indexes_exist()
    
    # Get all PDB IDs first
    pdb_ids = get_all_pdb_ids()
    print(f"Found {len(pdb_ids)} PDB entries to analyze")
    
    # Analyze maximum probability
    max_prob_metrics = analyze_max_probability(pdb_ids)
    plot_correlation_analysis(max_prob_metrics, 'max_probability')
    print_correlation_statistics(max_prob_metrics, 'max_probability')
    
    # Analyze mean probability
    mean_prob_metrics = analyze_mean_probability(pdb_ids)
    plot_correlation_analysis(mean_prob_metrics, 'mean_probability')
    print_correlation_statistics(mean_prob_metrics, 'mean_probability')
    
    # Analyze probability sum
    # sum_prob_metrics = analyze_probability_sum(pdb_ids)
    # plot_correlation_analysis(sum_prob_metrics, 'probability_sum')
    # print_correlation_statistics(sum_prob_metrics, 'probability_sum')
    
    # # Analyze top cluster mean
    # cluster_mean_metrics = analyze_top_cluster_mean(pdb_ids)
    # plot_correlation_analysis(cluster_mean_metrics, 'cluster_mean')
    # print_correlation_statistics(cluster_mean_metrics, 'cluster_mean')
    
    # # Analyze top 3 clusters mean
    # top_3_clusters_metrics = analyze_top_3_clusters_mean(pdb_ids)
    # plot_correlation_analysis(top_3_clusters_metrics, 'top_3_clusters_mean')
    # print_correlation_statistics(top_3_clusters_metrics, 'top_3_clusters_mean')
    
    return {
        'max_probability': max_prob_metrics,
        'mean_probability': mean_prob_metrics,
        # 'probability_sum': sum_prob_metrics,
        # 'cluster_mean': cluster_mean_metrics,
        # 'top_3_clusters_mean': top_3_clusters_metrics
    }

def analyze_binding_units():
    """Get all different binding types and units in the database."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get all binding data
        cur.execute("""
            SELECT DISTINCT binding_type, binding_unit
            FROM pdbbind
            WHERE binding_type IS NOT NULL
            ORDER BY binding_type, binding_unit
        """)
        results = cur.fetchall()
        
        print("\nBinding Types and Units:")
        print("------------------------")
        for binding_type, binding_unit in results:
            print(f"{binding_type}: {binding_unit}")
        
        # Get some example values for each type
        print("\nExample Values:")
        print("--------------")
        for binding_type, binding_unit in results:
            cur.execute("""
                SELECT pdb_id, binding_value, binding_type, binding_unit
                FROM pdbbind
                WHERE binding_type = %s AND binding_unit = %s
                LIMIT 3
            """, (binding_type, binding_unit))
            examples = cur.fetchall()
            print(f"\n{binding_type} ({binding_unit}):")
            for pdb_id, value, type_, unit in examples:
                print(f"  {pdb_id}: {value} {unit}")

def add_protein_structure_to_pdbbind_test():
    """Add protein structure columns to pdbbind_test table and populate them with parsed structure data."""
    # First add all necessary columns to the table
    columns = {
        'protein_pos': 'BYTEA',
        'protein_one_hot': 'BYTEA',
        'residue_ids': 'BYTEA',
        'atom_coords': 'BYTEA',
        'protein_backbone': 'BYTEA'
    }
    
    for column_name, column_type in columns.items():
        add_column('pdbbind_test', column_name, column_type)
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # First get all PDB IDs that need processing
        cur.execute("""
            SELECT DISTINCT p.pdb_id
            FROM pdbbind p
            LEFT JOIN pdbbind_test pt ON p.pdb_id = pt.pdb_id
            WHERE pt.protein_pos IS NULL 
               OR pt.protein_one_hot IS NULL 
               OR pt.residue_ids IS NULL 
               OR pt.atom_coords IS NULL 
               OR pt.protein_backbone IS NULL
            ORDER BY p.pdb_id
        """)
        pdb_ids = [row[0] for row in cur.fetchall()]
        total_entries = len(pdb_ids)
        print(f"Total entries to process: {total_entries}")
        
        # Process each PDB ID
        for idx, pdb_id in enumerate(pdb_ids, 1):
            try:
                # Get PDB data for this ID
                cur.execute("""
                    SELECT receptor_pdb
                    FROM pdbbind
                    WHERE pdb_id = %s
                """, (pdb_id,))
                receptor_pdb = cur.fetchone()[0]
                
                # Convert binary data to string
                pdb_content = receptor_pdb.tobytes().decode('utf-8')
                
                # Parse the PDB content
                struct = Structure(io.StringIO(pdb_content), skip_hetatm=False, skip_water=True)
                
                # Get all structure data using existing function
                (protein_pos, protein_one_hot, residue_ids, 
                 atom_coords, protein_backbone) = parse_protein_structure(struct)
                
                # Store all data in database
                cur.execute("""
                    UPDATE pdbbind_test 
                    SET protein_pos = %s,
                        protein_one_hot = %s,
                        residue_ids = %s,
                        atom_coords = %s,
                        protein_backbone = %s
                    WHERE pdb_id = %s
                """, (
                    pickle.dumps(protein_pos),
                    pickle.dumps(protein_one_hot),
                    pickle.dumps(residue_ids),
                    pickle.dumps(atom_coords),
                    pickle.dumps(protein_backbone),
                    pdb_id
                ))
                
                conn.commit()
                print(f"[{idx}/{total_entries}] Successfully processed {pdb_id}")
                
            except Exception as e:
                print(f"[{idx}/{total_entries}] Error processing {pdb_id}: {str(e)}")
                continue
        
        print("\nFinished processing all entries.")

def cluster_pocket_predictions(pocket_pred, protein_pos, cutoff=0.1, distance_threshold=5.0):
    """Cluster pocket predictions based on spatial positions of residues.
    
    Args:
        pocket_pred: numpy array of predicted probabilities
        protein_pos: numpy array of protein CA atom positions with shape (n_residues, 3)
        cutoff: probability threshold for selecting residues to cluster (default: 0.1)
        distance_threshold: maximum distance between two samples to be considered as neighbors (in Angstroms)
    
    Returns:
        numpy array: cluster labels with same shape as pocket_pred
        -1: no cluster
        0, 1, 2...: cluster labels
    """
    # Truncate pocket_pred if it's longer than protein_pos
    if len(pocket_pred) > len(protein_pos):
        pocket_pred = pocket_pred[:len(protein_pos)]
    
    # Select residues to cluster based on probability
    mask = (pocket_pred > cutoff)
    
    # Get coordinates of selected residues
    selected_pos = protein_pos[mask]
    
    if len(selected_pos) == 0:
        print(f"Warning: No residues selected for clustering with cutoff {cutoff}")
        return np.full_like(pocket_pred, -1, dtype=int)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=distance_threshold, min_samples=3).fit(selected_pos)
    
    # Create output array
    clusters = np.full_like(pocket_pred, -1, dtype=int)  # Initialize with -1
    clusters[mask] = clustering.labels_  # DBSCAN labels start from 0, -1 for noise
    
    return clusters

def calculate_pocket_clusters():
    """Calculate and store pocket clusters for all entries in pdbbind_test table.
    For each entry:
    1. Get pocket prediction and protein positions
    2. Calculate clusters using DBSCAN
    3. Store cluster labels back to database
    """
    from tqdm import tqdm
    
    # Add clusters column if it doesn't exist
    add_column('pdbbind_test', 'clusters', 'BYTEA')
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # First get all PDB IDs that need processing
        cur.execute("""
            SELECT DISTINCT p.pdb_id
            FROM pdbbind p
            LEFT JOIN pdbbind_test pt ON p.pdb_id = pt.pdb_id
            WHERE pt.clusters IS NULL 
               AND pt.pocket_pred IS NOT NULL 
               AND pt.protein_pos IS NOT NULL
            ORDER BY p.pdb_id
        """)
        pdb_ids = [row[0] for row in cur.fetchall()]
        total_entries = len(pdb_ids)
        print(f"Total entries to process: {total_entries}")
        
        # Process each PDB ID with tqdm progress bar
        for pdb_id in tqdm(pdb_ids, desc="Calculating pocket clusters", total=total_entries):
            try:
                # Get pocket prediction and protein positions
                cur.execute("""
                    SELECT pocket_pred, protein_pos
                    FROM pdbbind_test
                    WHERE pdb_id = %s
                """, (pdb_id,))
                pocket_pred, protein_pos = cur.fetchone()
                
                # Unpickle data
                pocket_pred = pickle.loads(pocket_pred).squeeze()
                protein_pos = pickle.loads(protein_pos)
                
                # Calculate clusters
                clusters = cluster_pocket_predictions(
                    pocket_pred, 
                    protein_pos,
                    cutoff=0.05,  # Lower threshold to catch more potential pockets
                    distance_threshold=5.0  # 5 Angstroms is a reasonable distance for protein pockets
                )
                
                # Store clusters back to database
                cur.execute("""
                    UPDATE pdbbind_test 
                    SET clusters = %s
                    WHERE pdb_id = %s
                """, (pickle.dumps(clusters), pdb_id))
                
                conn.commit()
                
            except Exception as e:
                tqdm.write(f"Error processing {pdb_id}: {str(e)}")
                continue
        
        print("\nFinished processing all entries.")

if __name__ == "__main__":
    # create_pdbbind_test_table()
    # predict_pocket_for_pdbbind()
    # analyze_binding_units()
    # add_protein_structure_to_pdbbind_test()
    # calculate_pocket_clusters()
    analyze_binding_correlation()


# %%
