#%%

from sklearn.cluster import DBSCAN
import numpy as np
from io import StringIO
import sys
import tempfile
import os
sys.path.append('..')
from src.db_utils import db_select, db_connection
from src.pdb_utils import Structure  # Add this import
from scipy.spatial.distance import pdist, squareform
from contextlib import contextmanager

@contextmanager
def temp_file_manager(mode='w+', suffix=None):
    """Create a temporary file in the current directory that is automatically deleted when the context exits.
    
    Args:
        mode: file open mode (default: 'w+')
        suffix: optional file suffix (e.g., '.pdb')
    
    Yields:
        A file object that will be automatically closed and deleted when the context exits.
    """
    temp_path = None
    try:
        # Create temporary file in current directory
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, dir='.')
        os.close(temp_fd)  # Close the file descriptor
        
        # Open the file with the requested mode
        file = open(temp_path, mode)
        yield file
        
        # Close the file if it's still open
        if not file.closed:
            file.close()
            
    finally:
        # Delete the temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_path}: {e}")

def cluster_pocket_predictions_ca(pocket_pred, protein_pos, cutoff=0.07, distance_threshold=5.0):
    """Cluster pocket predictions based on spatial positions of residues.
    
    Args:
        pocket_pred: numpy array of predicted probabilities
        protein_pos: numpy array of protein CA atom positions with shape (n_residues, 3)
        cutoff: probability threshold for selecting residues to cluster (default: 0.1)
    
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
    # eps: maximum distance between two samples to be considered as neighbors (in Angstroms)
    # min_samples: minimum number of samples in a neighborhood to be considered a core point
    clustering = DBSCAN(eps=distance_threshold, min_samples=3).fit(selected_pos)
    
    # Create output array
    clusters = np.full_like(pocket_pred, -1, dtype=int)  # Initialize with -1
    clusters[mask] = clustering.labels_  # DBSCAN labels start from 0, -1 for noise
    
    return clusters

def calculate_residue_min_distance(res1, res2):
    """Calculate minimum distance between all atoms of two residues.
    
    Args:
        res1: First residue object
        res2: Second residue object
    
    Returns:
        float: Minimum distance between any two atoms of the residues
    """
    atoms1 = res1.get_atoms()
    atoms2 = res2.get_atoms()
    
    min_dist = float('inf')
    for atom1 in atoms1:
        coord1 = atom1.get_coord()
        for atom2 in atoms2:
            coord2 = atom2.get_coord()
            dist = np.linalg.norm(coord1 - coord2)
            if dist < min_dist:
                min_dist = dist
    
    return min_dist

def cluster_pocket_predictions(pocket_pred, structure, cutoff=0.07, distance_threshold=3.0):
    """Cluster pocket predictions based on minimum distances between all atoms of residues.
    
    Args:
        pocket_pred: numpy array of predicted probabilities
        structure: Structure object containing protein coordinates
        cutoff: probability threshold for selecting residues to cluster (default: 0.07)
        distance_threshold: maximum distance between residues to be considered neighbors (default: 5.0)
    
    Returns:
        numpy array: cluster labels with same shape as pocket_pred
        -1: no cluster
        0, 1, 2...: cluster labels
    """
    # Get residues from the first model
    residues = structure[0].get_residues()
    
    # Truncate pocket_pred if it's longer than number of residues
    if len(pocket_pred) > len(residues):
        pocket_pred = pocket_pred[:len(residues)]
    
    # Select residues to cluster based on probability
    mask = (pocket_pred > cutoff)
    selected_indices = np.where(mask)[0]
    
    if len(selected_indices) == 0:
        print(f"Warning: No residues selected for clustering with cutoff {cutoff}")
        return np.full_like(pocket_pred, -1, dtype=int)
    
    # Get selected residues
    selected_residues = [residues[i] for i in selected_indices]
    
    # print(f"Selected {len(selected_residues)} residues for clustering...")
    
    # Calculate pairwise minimum distances between selected residues
    n_selected = len(selected_residues)
    distance_matrix = np.zeros((n_selected, n_selected))
    
    for i in range(n_selected):
        for j in range(i + 1, n_selected):
            min_dist = calculate_residue_min_distance(selected_residues[i], selected_residues[j])
            distance_matrix[i, j] = min_dist
            distance_matrix[j, i] = min_dist  # Symmetric matrix
    
    # print(f"Calculated distance matrix of shape {distance_matrix.shape}")
    # print(f"Distance matrix range: {np.min(distance_matrix[distance_matrix > 0]):.2f} - {np.max(distance_matrix):.2f} Å")
    
    # Perform DBSCAN clustering using precomputed distances
    # eps: maximum distance between two samples to be considered as neighbors (in Angstroms)
    # min_samples: minimum number of samples in a neighborhood to be considered a core point
    clustering = DBSCAN(eps=distance_threshold, min_samples=3, metric='precomputed').fit(distance_matrix)
    
    # Create output array
    clusters = np.full_like(pocket_pred, -1, dtype=int)  # Initialize with -1
    clusters[mask] = clustering.labels_  # DBSCAN labels start from 0, -1 for noise
    
    return clusters

def write_bfactor_to_pdb(bfactors, pdb_content, output_file=None, virtual_atom=True):
    """Write values to PDB file as beta factors.
    Only residues with CA atoms are considered.
    Also adds a virtual atom with bfactor=1 that can be used for normalization.
    The virtual atom has altloc='B' to avoid conflicts with protein atoms.
    
    Args:
        bfactors: numpy array of values to write as beta factors
        pdb_content: PDB content as string
        output_file: optional file path to save the modified PDB
    
    Returns:
        str: Modified PDB content
    """
    # Parse PDB content using Structure class
    structure = Structure()
    structure.read(StringIO(pdb_content))
    
    # Write structure with bfactors to a temporary file first
    with temp_file_manager(mode='w+', suffix='.pdb') as temp_file:
        structure.write(temp_file.name, residue_bfactors=bfactors)
        temp_file.seek(0)
        content = temp_file.read()
        
        # Add virtual atom with bfactor=1 for normalization in PyMOL
        # Using HETATM record with special residue name "VT" that can be easily hidden
        # Note: altloc='B' is used to avoid conflicts with protein atoms
        if virtual_atom:
            virtual_atom = "HETATM 9999 N   C VT X 999       0.000   0.000   0.000  1.00  1.00\n"
            modified_pdb = virtual_atom + content
        else:
            modified_pdb = content
        
        # Save to file if output_file is provided
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write(modified_pdb)
        
        return modified_pdb

def plot_pocket_pred_boxplot(lname):
    """Create a box plot for pocket predictions of a specific ligand.
    
    Args:
        lname: ligand name (e.g., '1s3f_0')
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import to_rgba
    
    # Get pocket predictions and mask from database
    id = db_select('processed_datasets', 'id', f"lname = '{lname}'")
    pocket_pred = db_select('moad_test_results', 'pocket_pred', f"id = {id}", is_pickle=True).squeeze()
    protein_mask = db_select('processed_datasets', 'protein_mask', f"id = {id}", is_pickle=True).squeeze()
    
    # Convert mask to boolean and use it for indexing
    protein_mask = protein_mask.astype(bool)
    
    # Only use predictions within the mask
    masked_preds = pocket_pred[protein_mask]
    
    # Set custom colors
    edge_color = '#43a3ef'
    outlier_color = '#ef767b'
    # Create lighter version of the blue color for fill
    fill_color = to_rgba(edge_color, alpha=0.1)
    
    # Create figure
    plt.figure(figsize=(3.5, 2.5))
    sns.boxplot(data=masked_preds, 
                color='#ffffff',
                boxprops={'edgecolor': '#000000', 'linewidth': 1},
                whiskerprops={'color': '#000000', 'linewidth': 1},
                capprops={'color': '#000000', 'linewidth': 1},
                medianprops={'color': '#000000', 'linewidth': 1},
                flierprops={'markerfacecolor': outlier_color, 
                           'markeredgecolor': 'none',
                           'alpha': 0.5})
    plt.ylabel('Prediction Value')
    
    # Print filename before saving
    filename = f'{lname}_pocket_pred_boxplot.svg'
    print(f"Saving figure to: {filename}")
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def analyze_pocket_pred_outliers(lname):
    """Analyze outliers in pocket predictions using IQR method.
    Only analyzes predictions within the protein mask.
    
    Args:
        lname: ligand name (e.g., '1s3f_0')
    """
    # Get pocket predictions and mask from database
    id = db_select('processed_datasets', 'id', f"lname = '{lname}'")
    pocket_pred = db_select('moad_test_results', 'pocket_pred', f"id = {id}", is_pickle=True).squeeze()
    protein_mask = db_select('processed_datasets', 'protein_mask', f"id = {id}", is_pickle=True).squeeze()
    
    # Convert mask to boolean and use it for indexing
    protein_mask = protein_mask.astype(bool)
    
    # Only use predictions within the mask
    masked_preds = pocket_pred[protein_mask]
    
    # Get positive predictions only
    # positive_preds = masked_preds[masked_preds > 0]
    positive_preds = masked_preds
    
    # Calculate Q1, Q3 and IQR
    q1 = np.percentile(positive_preds, 25)
    q3 = np.percentile(positive_preds, 75)
    iqr = q3 - q1
    
    # Calculate threshold using Q3 + 1.5*IQR
    iqr_threshold = q3 + 1.5 * iqr
    
    # Find outliers
    outliers = positive_preds[positive_preds > iqr_threshold]
    
    print(f"\nOutlier Analysis for {lname} (Protein Mask):")
    print(f"Total predictions in protein mask: {len(masked_preds)}")
    print(f"Positive predictions in mask: {len(positive_preds)}")
    print(f"Q1: {q1:.3f}")
    print(f"Q3: {q3:.3f}")
    print(f"IQR: {iqr:.3f}")
    print(f"Outlier threshold (Q3 + 1.5*IQR): {iqr_threshold:.3f}")
    print(f"Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"Outlier values: {outliers}")
        print(f"Max prediction value: {np.max(outliers):.3f}")

def analyze_pocket_distribution(is_pocket):
    """Analyze the distribution of pocket vs non-pocket residues.
    
    Args:
        is_pocket: numpy array of ground truth labels (0 or 1)
        protein_mask: optional boolean mask to select only protein residues
    """
    # if protein_mask is not None:
    #     is_pocket = is_pocket[protein_mask==1]
    
    total = len(is_pocket)
    n_pocket = np.sum(is_pocket)
    pocket_ratio = n_pocket / total * 100
    
    print("\nPocket Distribution Analysis:")
    print(f"Total residues: {total}")
    print(f"Pocket residues: {n_pocket}")
    print(f"Pocket ratio: {pocket_ratio:.2f}%")
    print(f"Random prediction AP would be approximately: {pocket_ratio/100:.3f}")

def get_structure_from_db(lname):
    """Get Structure object from database.
    
    Args:
        lname: ligand name (e.g., '1s3f_0')
    
    Returns:
        Structure: Structure object containing protein coordinates
    """
    protein_name = lname[:4]  # Extract protein name from ligand name
    pdb_content = db_select('proteins', 'pdb', f"name = '{protein_name}'", is_bytea=True)
    
    structure = Structure()
    structure.read(StringIO(pdb_content))
    
    return structure

def test_clustering():
    # Randomly select a ligand name from processed_datasets
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT lname 
            FROM processed_datasets
            WHERE split = 'test'
            ORDER BY RANDOM()
            LIMIT 1
        """)
        lname = cur.fetchone()[0]

    # lname = '2vcj_0'
    # lname = '1f0s_0'
    # lname = '1d1v_0'
    lname = '1qjw_6'
    # lname = '1hi5_0'
    # lname = '2lig_0'
    # lname = '3fyg_25'
    # lname = '1dy4_1'
    # lname = '1o23_3'
    # lname = '1iv2_1'
    # lname = '1s3f_0'
    cutoff = 0.1
    distance_threshold = 5.0
    
    id = db_select('processed_datasets', 'id', f"lname = '{lname}'")
    protein_mask = db_select('processed_datasets', 'protein_mask', f"id = {id}", is_pickle=True).squeeze()
    pocket_pred = db_select('moad_test_results', 'pocket_pred', f"id = {id}", is_pickle=True).squeeze()
    protein_pos = db_select('raw_datasets', 'protein_pos', f"ligand_name = '{lname}'", is_pickle=True)
    is_pocket = db_select('raw_datasets', 'is_pocket', f"ligand_name = '{lname}'", is_pickle=True)
    print(f"Testing clustering for ligand: {lname}")
    print(protein_mask.shape)
    print(protein_pos.shape)
    print(pocket_pred.shape)
    
    # Analyze pocket distribution
    analyze_pocket_distribution(is_pocket)
    
    # Create box plot and analyze outliers
    plot_pocket_pred_boxplot(lname)
    analyze_pocket_pred_outliers(lname)
    
    # Test original clustering method (CA-based)
    print("\n=== Testing CA-based clustering ===")
    clusters_ca = cluster_pocket_predictions_ca(pocket_pred, protein_pos, cutoff=cutoff, distance_threshold=distance_threshold)
    print(f"CA-based clustering found {len(np.unique(clusters_ca[clusters_ca != -1]))} clusters")
    
    # Test new clustering method (structure-based with all atom distances)
    print("\n=== Testing structure-based clustering ===")
    structure = get_structure_from_db(lname)
    clusters_structure = cluster_pocket_predictions(pocket_pred, structure, cutoff=cutoff, distance_threshold=distance_threshold)
    print(f"Structure-based clustering found {len(np.unique(clusters_structure[clusters_structure != -1]))} clusters")
    
    # Get PDB content for writing output files
    pdb_content = db_select('proteins', 'pdb', f"name = '{lname[:4]}'", is_bytea=True)
    
    # Write clustering results to PDB files
    write_bfactor_to_pdb(clusters_ca+1, pdb_content, output_file=f'{lname}_clusters_ca.pdb')
    write_bfactor_to_pdb(clusters_structure+1, pdb_content, output_file=f'{lname}_clusters_structure.pdb')
    write_bfactor_to_pdb(pocket_pred, pdb_content, output_file=f'{lname}_pocket_pred.pdb')
    
    # Compare clustering results
    print(f"\n=== Clustering Comparison ===")
    print(f"CA-based clusters: {np.unique(clusters_ca, return_counts=True)}")
    print(f"Structure-based clusters: {np.unique(clusters_structure, return_counts=True)}")
    
    # Save MOL file for the ligand
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT mol
            FROM ligands
            WHERE name = %s AND protein_name = %s
        """, (lname, lname[:4]))
        mol_data = cursor.fetchone()
        if mol_data and mol_data[0] is not None:
            with open(f'{lname}.mol', 'wb') as f:
                f.write(mol_data[0].tobytes())
            print(f"Saved MOL file for {lname}")
        else:
            print(f"No MOL file found for {lname}")

if __name__ == "__main__":
    test_clustering()

# %%
