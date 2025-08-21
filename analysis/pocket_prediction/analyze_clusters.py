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


def get_ligand_positions_batch(lnames):
    """Get ligand atom positions for multiple ligands in a single database query.
    
    Args:
        lnames: list of ligand names
    
    Returns:
        dict: mapping from ligand name to numpy array of positions
    """
    if not lnames:
        return {}
    
    ligand_positions_cache = {}
    with db_connection() as conn:
        cur = conn.cursor()
        # Use placeholders for IN clause
        placeholders = ','.join(['%s'] * len(lnames))
        cur.execute(f"""
            SELECT ligand_name, molecule_pos 
            FROM raw_datasets 
            WHERE ligand_name IN ({placeholders})
        """, lnames)
        
        for lname, molecule_pos in cur.fetchall():
            ligand_positions_cache[lname] = pickle.loads(molecule_pos)
    
    # Check for missing ligands
    missing_ligands = set(lnames) - set(ligand_positions_cache.keys())
    if missing_ligands:
        print(f"Warning: Missing data for {len(missing_ligands)} ligands: {list(missing_ligands)[:5]}...")
    
    return ligand_positions_cache


def get_cluster_statistics():
    """Get cluster statistics from database for predicted, true, and all pockets.
    
    Returns:
        tuple: (pred_sizes, true_sizes, pred_nclusters, true_nclusters, all_nclusters)
            - pred_sizes: list of sizes for each predicted cluster
            - true_sizes: list of sizes for each true cluster
            - pred_nclusters: list of number of clusters per protein for predictions
            - true_nclusters: list of number of clusters per protein for true pockets
            - all_nclusters: list of number of clusters per protein for all pockets combined
    """
    # Get all results from database
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get predicted and true clusters
        cur.execute("""
            SELECT m.clusters, m.true_clusters, p.all_clusters
            FROM moad_test_results m
            JOIN processed_datasets pd ON m.id = pd.id
            JOIN proteins p ON pd.pname = p.name
            WHERE m.clusters IS NOT NULL 
            AND m.true_clusters IS NOT NULL
            AND p.all_clusters IS NOT NULL
        """)
        results = cur.fetchall()
    
    print(f"Processing {len(results)} proteins")
    
    # Lists to store data
    pred_sizes = []
    true_sizes = []
    pred_nclusters = []
    true_nclusters = []
    all_nclusters = []
    
    # Process each protein
    for clusters, true_clusters, all_clusters in tqdm(results, desc="Analyzing clusters"):
        try:
            # Unpickle data
            clusters = pickle.loads(clusters)
            true_clusters = pickle.loads(true_clusters)
            all_clusters = pickle.loads(all_clusters)
            
            # Get unique clusters (excluding noise which is -1)
            pred_unique = np.unique(clusters)
            pred_unique = pred_unique[pred_unique != -1]
            
            true_unique = np.unique(true_clusters)
            true_unique = true_unique[true_unique != -1]
            
            all_unique = np.unique(all_clusters)
            all_unique = all_unique[all_unique != -1]
            
            # Store number of clusters
            pred_nclusters.append(len(pred_unique))
            true_nclusters.append(len(true_unique))
            all_nclusters.append(len(all_unique))
            
            # Calculate size of each cluster
            for cluster_id in pred_unique:
                cluster_size = np.sum(clusters == cluster_id)
                pred_sizes.append(cluster_size)
                
            for cluster_id in true_unique:
                cluster_size = np.sum(true_clusters == cluster_id)
                true_sizes.append(cluster_size)
                
        except Exception as e:
            print(f"Error processing a protein: {str(e)}")
            continue
    
    return (np.array(pred_sizes), np.array(true_sizes), 
            np.array(pred_nclusters), np.array(true_nclusters), np.array(all_nclusters))

def print_statistics(name, data):
    """Print statistics for a dataset."""
    print(f"\n{name}:")
    print(f"Count: {len(data)}")
    print(f"Mean: {np.mean(data):.1f}")
    print(f"Median: {np.median(data):.1f}")
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Std dev: {np.std(data):.1f}")

def plot_distribution(data_dict, xlabel, save_path, xlim, nbins):
    """Plot distribution of multiple datasets using line plots.
    
    Args:
        data_dict: Dictionary mapping labels to data arrays
        xlabel: Label for x-axis
        save_path: Path to save the plot
        xlim: Maximum x value
        nbins: Number of bins
    """
    plt.figure(figsize=(3.5, 2.5))
    
    bins = np.linspace(0, xlim, nbins)
    colors = {'Predicted': '#43a3ef', 'True': '#ef767b', 'All': '#65c366'}
    
    for label, data in data_dict.items():
        # Calculate histogram
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        # Get bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Plot line
        plt.plot(bin_centers, hist, label=label, color=colors[label], linewidth=2)
        # Add fill between line and x-axis
        plt.fill_between(bin_centers, hist, alpha=0.5, color=colors[label])
    
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.xlim(0, xlim)
    
    # Adjust legend style
    legend = plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path, bbox_inches='tight', format='svg')
    plt.show()
    plt.close()

def plot_pocket_statistics(save_dir='plots'):
    """Plot statistics for pocket clusters including size distribution and number of clusters."""
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all statistics
    pred_sizes, true_sizes, pred_nclusters, true_nclusters, all_nclusters = get_cluster_statistics()
    
    # Print statistics
    print("\nCluster Size Statistics:")
    print_statistics("Predicted Clusters", pred_sizes)
    print_statistics("True Clusters", true_sizes)
    
    print("\nNumber of Clusters per Protein Statistics:")
    print_statistics("Predicted", pred_nclusters)
    print_statistics("True", true_nclusters)
    print_statistics("All", all_nclusters)
    
    # Plot size distribution
    plot_distribution(
        {'Predicted': pred_sizes, 'True': true_sizes},
        'Pocket Size (number of residues)',
        os.path.join(save_dir, 'cluster_size_distribution.svg'),
        xlim=30, nbins=31
    )
    
    # Plot number of clusters distribution
    plot_distribution(
        {'Predicted': pred_nclusters, 'True': true_nclusters, 'All': all_nclusters},
        'Number of Pockets',
        os.path.join(save_dir, 'nclusters_distribution.svg'),
        xlim=20, nbins=21
    )

def get_cluster_predictions():
    """Get cluster predictions and ground truth data from database.
    
    Returns:
        list: List of tuples (clusters, pocket_pred, is_pocket, all_pockets, pname)
    """
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.clusters, m.pocket_pred, pd.is_pocket, p.all_pockets, pd.pname
            FROM moad_test_results m
            JOIN processed_datasets pd ON m.id = pd.id
            JOIN proteins p ON pd.pname = p.name
            WHERE m.clusters IS NOT NULL 
            AND pd.is_pocket IS NOT NULL 
            AND p.all_pockets IS NOT NULL 
            AND m.pocket_pred IS NOT NULL
        """)
        return cur.fetchall()

def process_protein_data(data):
    """Process raw protein data from database.
    
    Args:
        data: Tuple of (clusters, pocket_pred, is_pocket, all_pockets, pname)
        
    Returns:
        tuple: (clusters, pocket_pred, is_pocket, all_pockets, cluster_probs)
            where cluster_probs is list of (cluster_id, probability) sorted by probability
        or None if processing fails
    """
    clusters, pocket_pred, is_pocket, all_pockets, _ = data
    try:
        # Convert arrays from pickle
        clusters = pickle.loads(clusters)
        pocket_pred = pickle.loads(pocket_pred).squeeze()
        is_pocket = pickle.loads(is_pocket)
        all_pockets = pickle.loads(all_pockets)
        
        # Truncate arrays to minimum length silently
        min_length = min(len(clusters), len(pocket_pred), len(is_pocket), len(all_pockets))
        clusters = clusters[:min_length]
        pocket_pred = pocket_pred[:min_length]
        is_pocket = is_pocket[:min_length]
        all_pockets = all_pockets[:min_length]
        
        # Skip if no clusters predicted
        if len(clusters) == 0:
            return None
            
        # Get unique clusters (excluding -1 which is noise)
        unique_clusters = np.unique(clusters)
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        if len(unique_clusters) == 0:
            return None
            
        # For each cluster, get its maximum probability
        cluster_probs = []
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            max_prob = np.max(pocket_pred[cluster_mask])
            cluster_probs.append((cluster_id, max_prob))
        
        # Sort clusters by probability (highest first)
        cluster_probs.sort(key=lambda x: x[1], reverse=True)
        
        return clusters, pocket_pred, is_pocket, all_pockets, cluster_probs
        
    except Exception as e:
        print(f"Error processing protein data: {str(e)}")
        return None

def is_prediction_correct(clusters, is_pocket, all_pockets, cluster_probs, k=None, ratio_threshold=0.5):
    """Evaluate if a single prediction is correct.
    
    Args:
        clusters: Array of cluster assignments
        is_pocket: Array of ligand-specific pocket labels
        all_pockets: Array of all known pocket labels
        cluster_probs: List of (cluster_id, probability) sorted by probability
        k: Number of top clusters to check. If None, check top 3 clusters.
        ratio_threshold: Minimum ratio of true pocket residues required (default: 0.5)
    
    Returns:
        tuple: (is_pocket_correct, all_pockets_correct)
    """
    is_pocket_success = False
    all_pockets_success = False
    
    # Determine which clusters to check
    n_clusters = len(cluster_probs) if k is None else min(k, len(cluster_probs))
    check_clusters = cluster_probs[:n_clusters]
    
    # Check each cluster
    for cluster_id, _ in check_clusters:
        cluster_mask = clusters == cluster_id
        true_ratio_is_pocket = np.mean(is_pocket[cluster_mask])
        true_ratio_all_pockets = np.mean(all_pockets[cluster_mask])
        
        if true_ratio_is_pocket >= ratio_threshold:
            is_pocket_success = True
        if true_ratio_all_pockets >= ratio_threshold:
            all_pockets_success = True
    
    return is_pocket_success, all_pockets_success

def calculate_true_positives(results, evaluation_values, evaluation_type='top_k'):
    """Calculate true positives for different evaluation values.
    
    Args:
        results: List of database results
        evaluation_values: List of values to evaluate against (k values or ratio thresholds)
        evaluation_type: Either 'top_k' or 'ratio_threshold'
    
    Returns:
        tuple: (total_predictions, true_positives_dict)
    """
    total_predictions = 0
    true_positives = {
        'is_pocket': {v: 0 for v in evaluation_values},
        'all_pockets': {v: 0 for v in evaluation_values}
    }
    
    for data in tqdm(results, desc="Processing predictions"):
        processed_data = process_protein_data(data)
        if processed_data is None:
            continue
            
        clusters, _, is_pocket, all_pockets, cluster_probs = processed_data
        total_predictions += 1
        
        # Check each evaluation value
        for value in evaluation_values:
            if evaluation_type == 'top_k':
                is_pocket_correct, all_pockets_correct = is_prediction_correct(
                    clusters, is_pocket, all_pockets, cluster_probs, 
                    k=value, ratio_threshold=0.5
                )
            else:  # ratio_threshold
                is_pocket_correct, all_pockets_correct = is_prediction_correct(
                    clusters, is_pocket, all_pockets, cluster_probs,
                    k=3, ratio_threshold=value
                )
            
            if is_pocket_correct:
                true_positives['is_pocket'][value] += 1
            if all_pockets_correct:
                true_positives['all_pockets'][value] += 1
    
    return total_predictions, true_positives

def plot_precision_curve(save_dir='plots'):
    """Calculate and plot precision for different numbers of top predictions."""
    print("Calculating precision for different numbers of top predictions...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    results = get_cluster_predictions()
    print(f"Processing {len(results)} proteins")
    
    # Calculate true positives for different k values
    k_values = list(range(1, 16))  # 1 to 15
    total_predictions, true_positives = calculate_true_positives(
        results, k_values, evaluation_type='top_k'
    )
    
    # Calculate and plot precisions
    plot_precision_results(true_positives, total_predictions, k_values, 
                         'Top k', 'precision_comparison.svg', save_dir)

def plot_precision_by_ratio(save_dir='plots'):
    """Calculate and plot precision at different true pocket residue ratios for top 3 predictions."""
    print("Analyzing precision at different true pocket residue ratios...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    results = get_cluster_predictions()
    print(f"Processing {len(results)} proteins")
    
    # Calculate true positives for different ratio thresholds
    ratio_thresholds = np.arange(0.1, 1.1, 0.1)
    total_predictions, true_positives = calculate_true_positives(
        results, ratio_thresholds, evaluation_type='ratio_threshold'
    )
    
    # Calculate and plot precisions
    plot_precision_results(true_positives, total_predictions, ratio_thresholds, 
                         'Ratio threshold', 'precision_by_ratio.svg', save_dir)

def plot_precision_results(true_positives, total_predictions, x_values, xlabel, filename, save_dir):
    """Plot precision results.
    
    Args:
        true_positives: Dict of criterion -> value -> count
        total_predictions: Total number of predictions
        x_values: Values for x-axis
        xlabel: Label for x-axis
        filename: Output filename
        save_dir: Directory to save the plot
    """
    # Calculate precisions
    precisions = {
        'is_pocket': {},
        'all_pockets': {}
    }
    for criterion in ['is_pocket', 'all_pockets']:
        for x in x_values:
            precisions[criterion][x] = true_positives[criterion][x] / total_predictions if total_predictions > 0 else 0
    
    # Print statistics
    print(f"\nOverall Statistics:")
    print(f"Total number of predictions: {total_predictions}")
    
    for x in x_values:
        print(f"\nValue {x:.1f}:")
        for criterion in ['is_pocket', 'all_pockets']:
            print(f"Using {criterion} criterion:")
            print(f"Number of true positives: {true_positives[criterion][x]}")
            print(f"Precision: {precisions[criterion][x]:.3f}")
    
    # Create plot
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot lines for both criteria
    colors = {'is_pocket': '#43a3ef', 'all_pockets': '#ef767b'}
    labels = {
        'is_pocket': 'Ligand-specific pocket',
        'all_pockets': 'All known pockets'
    }
    for criterion in ['is_pocket', 'all_pockets']:
        values = [precisions[criterion][x] for x in x_values]
        plt.plot(x_values, values, '-o', color=colors[criterion], 
                label=labels[criterion], linewidth=2, markersize=4)
    
    plt.xlabel(xlabel)
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    
    # Set axis limits based on plot type
    if 'ratio' in filename:  # ratio threshold plot
        plt.xlim(0, 1)
    else:  # top k plot
        plt.ylim(0, 1)
    
    # Save plot
    output_file = os.path.join(save_dir, filename)
    print(f"\nSaving plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight', format='svg')
    plt.show()
    plt.close()

def plot_npockets_pred_vs_npockets_all(save_dir='plots'):
    """Plot scatter plot comparing number of predicted pockets vs actual pockets.
    
    Args:
        save_dir: Directory to save the plot
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get data from database
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.clusters, p.all_clusters, pd.pname
            FROM moad_test_results m
            JOIN processed_datasets pd ON m.id = pd.id
            JOIN proteins p ON pd.pname = p.name
            WHERE m.clusters IS NOT NULL AND p.all_clusters IS NOT NULL
        """)
        results = cur.fetchall()
    
    print(f"Processing {len(results)} proteins")
    
    # Lists to store data
    pred_nclusters = []
    all_nclusters = []
    
    # Process each protein
    for clusters, all_clusters, pname in results:
        try:
            # Unpickle data
            clusters = pickle.loads(clusters)
            all_clusters = pickle.loads(all_clusters)
            
            # Get unique clusters (excluding noise which is -1)
            pred_unique = np.unique(clusters)
            pred_unique = pred_unique[pred_unique != -1]
            
            all_unique = np.unique(all_clusters)
            all_unique = all_unique[all_unique != -1]
            
            # Store number of clusters
            pred_nclusters.append(len(pred_unique))
            all_nclusters.append(len(all_unique))
                
        except Exception as e:
            print(f"Error processing protein {pname}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    pred_nclusters = np.array(pred_nclusters)
    all_nclusters = np.array(all_nclusters)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(pred_nclusters, all_nclusters)[0,1]
    
    # Create scatter plot
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot y=x line
    max_n = max(np.max(pred_nclusters), np.max(all_nclusters))
    plt.plot([0, max_n], [0, max_n], 'k--', alpha=0.5, label='y=x')
    
    # Plot scatter points
    plt.scatter(all_nclusters, pred_nclusters, alpha=0.5, color='#43a3ef')
    
    plt.xlabel('Number of Actual Pockets')
    plt.ylabel('Number of Predicted Pockets')
    plt.title(f'R = {correlation:.3f}')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_file = os.path.join(save_dir, 'npockets_correlation.svg')
    print(f"\nSaving plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print statistics
    print("\nStatistics:")
    print(f"Number of proteins analyzed: {len(pred_nclusters)}")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Mean predicted pockets: {np.mean(pred_nclusters):.1f}")
    print(f"Mean actual pockets: {np.mean(all_nclusters):.1f}")
    print(f"Median predicted pockets: {np.median(pred_nclusters):.1f}")
    print(f"Median actual pockets: {np.median(all_nclusters):.1f}")

def store_center_ligand_distances():
    """Calculate and store minimum distances between pocket centers and ligands in database."""
    print("Calculating and storing center-ligand distances...")
    
    # Add columns for storing distances if they don't exist
    add_column('moad_test_results', 'center_ligand_distances', 'BYTEA')
    add_column('moad_test_results', 'center_ligand_center_distances', 'BYTEA')
    
    # Get all data first
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.id, m.pocket_centers, m.center_scores, m.num_ligands,
                   pd.lname
            FROM moad_test_results m
            JOIN processed_datasets pd ON m.id = pd.id
            WHERE m.pocket_centers IS NOT NULL 
            AND m.center_scores IS NOT NULL
            AND m.num_ligands IS NOT NULL
        """)
        all_results = cur.fetchall()
    
    # Get all unique ligand names and batch fetch their positions
    unique_lnames = list(set([result[4] for result in all_results]))
    print(f"Found {len(unique_lnames)} unique ligands")
    
    ligand_positions_cache = get_ligand_positions_batch(unique_lnames)
    print(f"Cached {len(ligand_positions_cache)} ligand positions")
    
    print(f"Processing {len(all_results)} proteins")
    
    # Process proteins in batches
    batch_size = 100
    current_batch = []
    
    # Process each protein
    for id, centers, scores, num_ligands, lname in tqdm(all_results, desc="Calculating distances"):
        try:
            # Unpickle data
            centers = pickle.loads(centers)
            scores = pickle.loads(scores)
            
            # Skip if no centers found
            if len(centers) == 0:
                continue
                
            # Get ligand positions from cache
            if lname not in ligand_positions_cache:
                print(f"Warning: No cached data for ligand {lname}, skipping")
                continue
            ligand_pos = ligand_positions_cache[lname]
            
            # Calculate ligand center
            ligand_center = np.mean(ligand_pos, axis=0)
                        
            # Calculate minimum distance between each center and any ligand atom
            min_distances = []
            center_distances = []
            for center in centers:
                # Calculate distances to all ligand atoms
                distances = np.linalg.norm(ligand_pos - center, axis=1)
                min_distances.append(np.min(distances))
                
                # Calculate distance to ligand center
                center_dist = np.linalg.norm(ligand_center - center)
                center_distances.append(center_dist)
            
            # Store both types of distances
            min_distances = np.array(min_distances)
            center_distances = np.array(center_distances)
            current_batch.append((pickle.dumps(min_distances), pickle.dumps(center_distances), id))
            
            # Update database when batch is full
            if len(current_batch) >= batch_size:
                with db_connection() as conn:
                    cur = conn.cursor()
                    cur.executemany("""
                        UPDATE moad_test_results 
                        SET center_ligand_distances = %s,
                            center_ligand_center_distances = %s
                        WHERE id = %s
                    """, current_batch)
                    conn.commit()
                current_batch = []
                
        except Exception as e:
            print(f"Error processing protein {lname} (id: {id}): {str(e)}")
            raise e
    
    # Update remaining proteins
    if current_batch:
        with db_connection() as conn:
            cur = conn.cursor()
            cur.executemany("""
                UPDATE moad_test_results 
                SET center_ligand_distances = %s,
                    center_ligand_center_distances = %s
                WHERE id = %s
            """, current_batch)
            conn.commit()
    
    print("Finished storing distances")

def plot_center_ligand_distance():
    """Plot success rate of pocket center prediction based on distance to ligand using stored distances."""
    print("Analyzing pocket center distances to ligands...")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Get data from database
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.center_ligand_distances, m.pocket_centers, m.center_scores, m.num_ligands
            FROM moad_test_results m
            WHERE m.center_ligand_distances IS NOT NULL
            AND m.pocket_centers IS NOT NULL
            AND m.center_scores IS NOT NULL
            AND m.num_ligands IS NOT NULL
        """)
        results = cur.fetchall()
    
    print(f"Processing {len(results)} proteins")
    
    # Cutoff values from 0 to 20 Å
    cutoffs = np.arange(0, 20.1, 0.5)
    
    # Initialize success counts for each strategy
    success_counts_all = np.zeros_like(cutoffs)    # k = num_centers
    success_counts_fixed5 = np.zeros_like(cutoffs)  # k = 5
    success_counts_fixed3 = np.zeros_like(cutoffs)  # k = 3
    success_counts_fixed1 = np.zeros_like(cutoffs)  # k = 1
    total_proteins = 0
    
    # Process each protein
    for distances_pickle, centers_pickle, scores_pickle, num_ligands in tqdm(results, desc="Analyzing distances"):
        try:
            # Unpickle data
            all_distances = pickle.loads(distances_pickle)  # These are already sorted by score
            centers = pickle.loads(centers_pickle)
            
            if len(all_distances) == 0:
                continue
            
            # Get minimum distances for different k values
            k_all = len(all_distances)
            k_fixed5 = min(5, len(all_distances))  # Fixed k = 5
            k_fixed3 = min(3, len(all_distances))  # Fixed k = 3
            k_fixed1 = min(1, len(all_distances))  # Fixed k = 1
            
            # Calculate minimum distances for each strategy
            min_dist_all = np.min(all_distances)
            min_dist_fixed5 = np.min(all_distances[:k_fixed5])
            min_dist_fixed3 = np.min(all_distances[:k_fixed3])
            min_dist_fixed1 = np.min(all_distances[:k_fixed1])
            
            # Update success counts for each cutoff
            success_counts_all += (min_dist_all <= cutoffs)
            success_counts_fixed5 += (min_dist_fixed5 <= cutoffs)
            success_counts_fixed3 += (min_dist_fixed3 <= cutoffs)
            success_counts_fixed1 += (min_dist_fixed1 <= cutoffs)
            total_proteins += 1
                
        except Exception as e:
            print(f"Error processing a protein: {str(e)}")
            continue
    
    # Calculate success rates
    if total_proteins > 0:
        success_rates_all = success_counts_all / total_proteins
        success_rates_fixed5 = success_counts_fixed5 / total_proteins
        success_rates_fixed3 = success_counts_fixed3 / total_proteins
        success_rates_fixed1 = success_counts_fixed1 / total_proteins
    else:
        success_rates_all = np.zeros_like(cutoffs)
        success_rates_fixed5 = np.zeros_like(cutoffs)
        success_rates_fixed3 = np.zeros_like(cutoffs)
        success_rates_fixed1 = np.zeros_like(cutoffs)
    
    # Create plot
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot lines for each strategy
    plt.plot(cutoffs, success_rates_all, '-', color='#65c366', linewidth=2, 
             label='k = all')
    plt.plot(cutoffs, success_rates_fixed5, '-', color='#9370db', linewidth=2, 
             label='k = 5')
    plt.plot(cutoffs, success_rates_fixed3, '-', color='#ffa500', linewidth=2, 
             label='k = 3')
    plt.plot(cutoffs, success_rates_fixed1, '-', color='#20b2aa', linewidth=2, 
             label='k = 1')
    
    plt.xlabel('Distance Cutoff (Å)')
    plt.ylabel('Success Rate')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    
    # Save plot
    output_file = os.path.join('plots', 'center_ligand_distance.svg')
    print(f"\nSaving plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight', format='svg')
    plt.show()
    plt.close()
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total proteins analyzed: {total_proteins}")
    print("\nSuccess rates at selected cutoffs:")
    for cutoff, rate3, rate4, rate5, rate6 in zip(cutoffs[::4], 
                                               success_rates_all[::4],
                                               success_rates_fixed5[::4],
                                               success_rates_fixed3[::4],
                                               success_rates_fixed1[::4]):
        print(f"\nAt {cutoff:.1f} Å:")
        print(f"  k = all:   {rate3:.3f}")
        print(f"  k = 5:            {rate4:.3f}")
        print(f"  k = 3:            {rate5:.3f}")
        print(f"  k = 1:            {rate6:.3f}")

def find_failed_prediction_example(cutoff=10.0):
    """Find a random example where all predicted centers are far from ligands.
    
    Args:
        cutoff: Distance threshold in Angstroms (default: 10.0)
    """
    print(f"Finding a random example where all centers are farther than {cutoff}Å from ligands...")
    
    # Get data from database
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.center_ligand_distances, m.pocket_centers, m.center_scores,
                   pd.lname, m.num_ligands
            FROM moad_test_results m
            JOIN processed_datasets pd ON m.id = pd.id
            WHERE m.center_ligand_distances IS NOT NULL
            AND m.pocket_centers IS NOT NULL
            AND m.center_scores IS NOT NULL
        """)
        results = cur.fetchall()
    
    # Find all qualifying examples
    failed_cases = []
    for distances_pickle, centers_pickle, scores_pickle, lname, num_ligands in results:
        try:
            # Unpickle data
            all_distances = pickle.loads(distances_pickle)
            centers = pickle.loads(centers_pickle)
            scores = pickle.loads(scores_pickle)
            
            if len(all_distances) == 0:
                continue
            
            # Check if all distances are greater than cutoff
            if np.min(all_distances) > cutoff:
                failed_cases.append({
                    'lname': lname,
                    'min_distance': np.min(all_distances),
                    'num_centers': len(centers),
                    'num_ligands': num_ligands,
                    'center_scores': scores
                })
                
        except Exception as e:
            print(f"Error processing {lname}: {str(e)}")
            continue
    
    if failed_cases:
        # Randomly select one case
        import random
        case = random.choice(failed_cases)
        
        print("\nRandom failed prediction example:")
        print(f"Ligand name: {case['lname']}")
        print(f"Minimum distance to ligand: {case['min_distance']:.1f}Å")
        print(f"Number of predicted centers: {case['num_centers']}")
        print(f"Number of ligands in protein: {case['num_ligands']}")
        print("\nCenter scores:")
        for i, score in enumerate(case['center_scores']):
            print(f"  Center {i+1}: {score:.3f}")
    else:
        print(f"No cases found where all centers are farther than {cutoff}Å from ligands.")

def plot_center_ligand_distance_by_protein():
    """Plot success rate of pocket center prediction based on distance to ligand at protein level.
    
    Unlike plot_center_ligand_distance which calculates success rate for each protein-ligand pair,
    this function considers a protein successful if any of its ligands is correctly predicted.
    Creates two plots: one for distances to nearest ligand atoms, one for distances to ligand centers.
    """
    print("Analyzing pocket center distances to ligands at protein level...")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Get data from database, including protein name
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.center_ligand_distances, m.pocket_centers, m.center_scores, 
                   m.num_ligands, pd.pname
            FROM moad_test_results m
            JOIN processed_datasets pd ON m.id = pd.id
            WHERE m.center_ligand_distances IS NOT NULL
            AND m.pocket_centers IS NOT NULL
            AND m.center_scores IS NOT NULL
            AND m.num_ligands IS NOT NULL
        """)
        results = cur.fetchall()
    
    print(f"Processing {len(results)} protein-ligand pairs")
    
    # Cutoff values from 0 to 20 Å
    cutoffs = np.arange(0, 20.1, 0.5)
    
    # Group results by protein name
    protein_results = {}
    for distances_pickle, centers_pickle, scores_pickle, num_ligands, pname in results:
        if pname not in protein_results:
            protein_results[pname] = []
        protein_results[pname].append((distances_pickle, centers_pickle, scores_pickle, num_ligands))
    
    print(f"Found {len(protein_results)} unique proteins")
    
    # Initialize success counts for distances to nearest ligand atoms
    success_counts_all_atom = np.zeros_like(cutoffs)    # k = num_centers
    success_counts_fixed5_atom = np.zeros_like(cutoffs)  # k = 5
    success_counts_fixed3_atom = np.zeros_like(cutoffs)  # k = 3
    success_counts_fixed1_atom = np.zeros_like(cutoffs)  # k = 1
    
    total_proteins = 0
    
    # Process each protein
    for pname, protein_pairs in tqdm(protein_results.items(), desc="Analyzing proteins"):
        try:
            # For distances to nearest ligand atoms
            protein_success_all_atom = np.zeros_like(cutoffs, dtype=bool)
            protein_success_fixed5_atom = np.zeros_like(cutoffs, dtype=bool)
            protein_success_fixed3_atom = np.zeros_like(cutoffs, dtype=bool)
            protein_success_fixed1_atom = np.zeros_like(cutoffs, dtype=bool)
            
            # Check each ligand pair
            for distances_pickle, centers_pickle, scores_pickle, num_ligands in protein_pairs:
                # Unpickle data
                all_distances_atom = pickle.loads(distances_pickle)
                
                if len(all_distances_atom) == 0:
                    continue
                
                # Get minimum distances for different k values (atom distances)
                k_all = len(all_distances_atom)
                k_fixed5 = min(5, len(all_distances_atom))
                k_fixed3 = min(3, len(all_distances_atom))
                k_fixed1 = min(1, len(all_distances_atom))
                
                # Calculate minimum distances for each strategy (atom distances)
                min_dist_all_atom = np.min(all_distances_atom)
                min_dist_fixed5_atom = np.min(all_distances_atom[:k_fixed5])
                min_dist_fixed3_atom = np.min(all_distances_atom[:k_fixed3])
                min_dist_fixed1_atom = np.min(all_distances_atom[:k_fixed1])
                
                # Update protein success (using OR operation) - atom distances
                protein_success_all_atom |= (min_dist_all_atom <= cutoffs)
                protein_success_fixed5_atom |= (min_dist_fixed5_atom <= cutoffs)
                protein_success_fixed3_atom |= (min_dist_fixed3_atom <= cutoffs)
                protein_success_fixed1_atom |= (min_dist_fixed1_atom <= cutoffs)
            
            # Update success counts
            success_counts_all_atom += protein_success_all_atom
            success_counts_fixed5_atom += protein_success_fixed5_atom
            success_counts_fixed3_atom += protein_success_fixed3_atom
            success_counts_fixed1_atom += protein_success_fixed1_atom
            
            total_proteins += 1
                
        except Exception as e:
            print(f"Error processing protein {pname}: {str(e)}")
            continue
    
    # Calculate success rates
    if total_proteins > 0:
        success_rates_all_atom = success_counts_all_atom / total_proteins
        success_rates_fixed5_atom = success_counts_fixed5_atom / total_proteins
        success_rates_fixed3_atom = success_counts_fixed3_atom / total_proteins
        success_rates_fixed1_atom = success_counts_fixed1_atom / total_proteins
    else:
        success_rates_all_atom = np.zeros_like(cutoffs)
        success_rates_fixed5_atom = np.zeros_like(cutoffs)
        success_rates_fixed3_atom = np.zeros_like(cutoffs)
        success_rates_fixed1_atom = np.zeros_like(cutoffs)
    
    # Create plot: distances to nearest ligand atoms
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot lines for each strategy
    # plt.plot(cutoffs, success_rates_all_atom, '-', color='#65c366', linewidth=2, 
    #          label='k = all')
    plt.plot(cutoffs, success_rates_fixed5_atom, '-', color='black', linewidth=2, 
             label='DCC')
    plt.plot(cutoffs, success_rates_fixed3_atom, '-', color='#ef767b', linewidth=2, 
             label='N + 2')
    plt.plot(cutoffs, success_rates_fixed1_atom, '-', color='#43a3ef', linewidth=2, 
             label='N')
    
    plt.xlabel('Distance Cutoff (Å)')
    plt.ylabel('Success Rate')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    
    # Save plot
    output_file = os.path.join('plots', 'center_ligand_distance_by_protein.svg')
    print(f"\nSaving plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight', format='svg')
    plt.show()
    plt.close()
    
    # Save data as CSV
    import pandas as pd
    csv_data = {
        'Distance_Cutoff_A': cutoffs,
        'Success_Rate_k_all': success_rates_all_atom,
        'Success_Rate_k_5': success_rates_fixed5_atom,
        'Success_Rate_k_3': success_rates_fixed3_atom,
        'Success_Rate_k_1': success_rates_fixed1_atom
    }
    df = pd.DataFrame(csv_data)
    csv_file = os.path.join('plots', 'center_ligand_distance_by_protein.csv')
    print(f"Saving data to: {csv_file}")
    df.to_csv(csv_file, index=False)
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total unique proteins analyzed: {total_proteins}")
    
    print("\nSuccess rates at selected cutoffs:")
    for cutoff, rate_all, rate5, rate3, rate1 in zip(cutoffs[::4], 
                                                success_rates_all_atom[::4],
                                                success_rates_fixed5_atom[::4],
                                                success_rates_fixed3_atom[::4],
                                                success_rates_fixed1_atom[::4]):
        print(f"\nAt {cutoff:.1f} Å:")
        print(f"  k = all:   {rate_all:.3f}")
        print(f"  k = 5:     {rate5:.3f}")
        print(f"  k = 3:     {rate3:.3f}")
        print(f"  k = 1:     {rate1:.3f}")

if __name__ == "__main__":    
    # plot_pocket_statistics()
    # plot_npockets_pred_vs_npockets_all()
    # plot_precision_curve()
    # plot_precision_by_ratio()
    
    # store_center_ligand_distances()
    # plot_center_ligand_distance()
    plot_center_ligand_distance_by_protein()
    
    # find_failed_prediction_example()


    # Test performance optimization
    # import time
    
    # print("Testing individual get_ligand_positions calls:")
    # for lname in ['10gs_1', '10gs_2', '10gs_0']:
    #     t1 = time.time()
    #     get_ligand_positions(lname)
    #     t2 = time.time()
    #     print(f"  {lname}: {t2 - t1:.4f} seconds")
    
    # print("\nTesting batch approach:")
    # t1 = time.time()
    # batch_results = get_ligand_positions_batch(['10gs_1', '10gs_2', '10gs_0'])
    # t2 = time.time()
    # print(f"  Batch query: {t2 - t1:.4f} seconds")

# %%
