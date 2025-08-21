#%%

import numpy as np
from typing import List, Tuple, Optional
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pdb_utils import Structure, Atom
from src.clustering import write_bfactor_to_pdb
from scipy.fft import fftn, ifftn
from scipy import ndimage
import io
from src.db_utils import db_connection
import pickle
import time
from sklearn.cluster import DBSCAN

def save_points_as_pdb(output_file: str, points: np.ndarray, b_factors: Optional[np.ndarray] = None, 
                      atom_type: str = "O", residue_name: str = "SOL"):
    """Save points to a PDB file as HETATM records.
    
    Args:
        output_file: Path to output PDB file
        points: Array of shape (N, 3) containing point coordinates
        b_factors: Optional array of shape (N,) containing B-factor values
        atom_type: Atom type for the points (default: "O")
        residue_name: Residue name for the points (default: "SOL")
    """
    # Randomly sample 1000 points if there are more
    if len(points) > 1000:
        indices = np.random.choice(len(points), 1000, replace=False)
        sampled_points = points[indices]
        sampled_b_factors = b_factors[indices] if b_factors is not None else None
    else:
        sampled_points = points
        sampled_b_factors = b_factors
    
    with open(output_file, 'w') as f:
        # Write points as HETATM records
        for i, point in enumerate(sampled_points, 1):
            b_factor = f"{sampled_b_factors[i-1]:6.2f}" if sampled_b_factors is not None else "  0.00"
            hetatm = (
                f"HETATM{i:5d}  {atom_type}   {residue_name} X{i:4d}    "
                f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}"
                f"  1.00{b_factor}           {atom_type}  \n"
            )
            f.write(hetatm)
        
        f.write("END\n")

def select_random_with_pocket_pred(cur):
    """Select a random lname with pocket prediction data from database.
    
    Args:
        cur: Database cursor
        
    Returns:
        str: lname
    """
    cur.execute("""
        SELECT p.lname
        FROM moad_test_results m
        JOIN processed_datasets p ON m.id = p.id
        WHERE m.pocket_pred IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 1
    """)
    
    result = cur.fetchone()
    if result is None:
        raise ValueError("No entries found with pocket prediction data!")
    
    return result[0]

def get_pocket_pred_data(cur, lname):
    """Get pocket prediction data for a specific lname.
    
    Args:
        cur: Database cursor
        lname: Ligand name
        
    Returns:
        numpy.ndarray: pocket prediction data
    """
    cur.execute("""
        SELECT m.pocket_pred
        FROM moad_test_results m
        JOIN processed_datasets p ON m.id = p.id
        WHERE p.lname = %s
    """, (lname,))
    
    result = cur.fetchone()
    if result is None:
        raise ValueError(f"No pocket prediction data found for {lname}")
    
    return pickle.loads(result[0]).squeeze()

def get_protein_data(cur, lname):
    """Get protein PDB data from database.
    
    Args:
        cur: Database cursor
        lname: Ligand name (format: pname_ligand_id)
        
    Returns:
        tuple: (pdb_text, Structure object)
    """
    pname = lname.split('_')[0]  # Extract protein name from lname
    
    cur.execute("""
        SELECT pdb 
        FROM proteins 
        WHERE name = %s
    """, (pname,))
    
    pdb_result = cur.fetchone()
    if pdb_result is None:
        raise ValueError(f"No PDB data found for protein {pname}")
    
    pdb_text = pdb_result[0].tobytes().decode('utf-8')
    
    # Create structure object
    struct = Structure()
    struct.read(io.StringIO(pdb_text))
    
    return pdb_text, struct

def create_probe_kernel(probe_radius: float, grid_spacing: float, dimensions: Tuple[int, ...], value: float = 1.0) -> np.ndarray:
    """Create a probe kernel for pocket detection.
    
    Args:
        probe_radius: Radius of probe sphere in Angstroms
        grid_spacing: Spacing between grid points in Angstroms
        dimensions: Grid dimensions as tuple of integers
        value: Value to use for points within probe radius (default: 1.0)
        
    Returns:
        np.ndarray: Padded kernel array matching grid dimensions
    """
    probe_radius_grid = int(np.ceil(probe_radius / grid_spacing))
    
    x, y, z = np.mgrid[-probe_radius_grid:probe_radius_grid + 1,
                       -probe_radius_grid:probe_radius_grid + 1,
                       -probe_radius_grid:probe_radius_grid + 1]
    
    distances = np.sqrt(x**2 + y**2 + z**2) * grid_spacing
    
    # Create probe kernel with specified value within probe_radius
    kernel = np.where(distances <= probe_radius, value, 0.0).astype(np.float64)
    
    # Pad kernel to match grid size for FFT convolution
    kernel_padded = np.zeros(dimensions, dtype=np.float64)
    kernel_padded[:kernel.shape[0], :kernel.shape[1], :kernel.shape[2]] = kernel
    
    return kernel_padded

def calculate_score_with_fft(protein_grid: np.ndarray, kernel: np.ndarray, probe_radius: float, 
                         grid_spacing: float, dimensions: Tuple[int, ...]) -> np.ndarray:
    """Calculate scoring grid using FFT convolution.
    
    Args:
        protein_grid: Grid containing protein and pocket prediction values
        kernel: Probe kernel for convolution
        probe_radius: Radius of probe sphere in Angstroms
        grid_spacing: Spacing between grid points in Angstroms
        dimensions: Grid dimensions
        
    Returns:
        np.ndarray: Scoring grid after FFT convolution
    """
    # Perform FFT convolution
    protein_fft = np.array(fftn(protein_grid))
    kernel_fft = np.array(fftn(kernel))
    
    # Element-wise multiplication in frequency domain
    convolution_fft = protein_fft * kernel_fft
    
    # Transform back to spatial domain
    scoring_grid_ = np.real(np.array(ifftn(convolution_fft)))
    scoring_grid = np.zeros(protein_grid.shape, dtype=np.float64)
    probe_radius_grid = int(np.ceil(probe_radius / grid_spacing))
    scoring_grid[:dimensions[0]-probe_radius_grid, :dimensions[1]-probe_radius_grid, :dimensions[2]-probe_radius_grid] = scoring_grid_[probe_radius_grid:, probe_radius_grid:, probe_radius_grid:]
    
    return scoring_grid

def find_pocket_centers(structure: Structure, pocket_pred: np.ndarray, prob_threshold: float = 0.03,
                                     grid_spacing: float = 0.5, probe_radius: float = 2,
                                     inner_radius: float = 1.5, outer_radius: float = 2,
                                     top_k_points: int = 8000) -> dict:
    """Find pocket centers using FFT-based docking-like scoring approach.
    
    Args:
        structure: Structure object containing protein coordinates
        pocket_pred: Array of pocket predictions (same length as residues)
        grid_spacing: Spacing between grid points in Angstroms
        probe_radius: Radius of probe sphere in Angstroms (default 3.0A)
        inner_radius: Inner radius for repulsion zone (default 1.5A)
        outer_radius: Outer radius for attraction zone (default 2.5A)
        cluster_distance: Maximum distance between points to be considered in same cluster
        top_k_points: Number of top-scoring points to consider for clustering
        
    Returns:
        dict: Dictionary containing:
            - centers: Array of shape (n_clusters, 3) containing pocket center coordinates
            - center_scores: Array of shape (n_clusters,) containing scores1 for cluster centers
            - top_points: Array of shape (n_top, 3) containing top scoring points
            - top_scores2: Array of shape (n_top,) containing scores2 for top points
            - positive_points: Array of shape (n_positive, 3) containing all positive scoring points
            - positive_scores2: Array of shape (n_positive,) containing scores2 for positive points
    """

    atoms, coords, atom_to_residue, residue_list = [], [], [], []
    res_idx = 0
    for chain in structure[0]:
        for res in chain:
            residue_list.append(res)
            for atom in res:
                atoms.append(atom)
                coords.append(atom.get_coord())
                atom_to_residue.append(res_idx)
            res_idx += 1
    coords = np.array(coords)
    atom_to_residue = np.array(atom_to_residue)
    
    # Define grid boundaries with padding
    padding = 15.0  # Add padding around protein
    min_coords = np.min(coords, axis=0) - padding
    max_coords = np.max(coords, axis=0) + padding
    
    # Calculate grid dimensions
    dimensions = np.ceil((max_coords - min_coords) / grid_spacing).astype(int)
    
    # Initialize 3D grid for protein atoms
    protein_grid = np.zeros(dimensions, dtype=np.float64)
    
    # Convert atom coordinates to grid indices
    grid_coords = ((coords - min_coords) / grid_spacing).astype(int)
    
    # Place atoms in grid with their pocket prediction scores and van der Waals radii
    for i, (atom, idx, res_idx) in enumerate(zip(atoms, grid_coords, atom_to_residue)):
        if np.all(idx >= 0) and np.all(idx < dimensions):
            # Get pocket prediction for this residue
            if res_idx < len(pocket_pred):
                pocket_score = pocket_pred[res_idx]
            else:
                pocket_score = 0.0
            pocket_score = pocket_score if pocket_score > prob_threshold else 0.0
            
            vdw_grid_radius = int(np.ceil(outer_radius / grid_spacing))
            
            # Create a small grid around the atom
            x_range = slice(max(0, idx[0] - vdw_grid_radius), min(dimensions[0], idx[0] + vdw_grid_radius + 1))
            y_range = slice(max(0, idx[1] - vdw_grid_radius), min(dimensions[1], idx[1] + vdw_grid_radius + 1))
            z_range = slice(max(0, idx[2] - vdw_grid_radius), min(dimensions[2], idx[2] + vdw_grid_radius + 1))
            
            # Get grid coordinates for this region
            x_grid, y_grid, z_grid = np.mgrid[x_range, y_range, z_range]
            
            # Calculate distances in grid units
            distances = np.sqrt((x_grid - idx[0])**2 + (y_grid - idx[1])**2 + (z_grid - idx[2])**2) * grid_spacing
            
            # Mark grid points with different values based on distance from atom center
            # Inner shell (0 to inner_radius): negative values (repulsion)
            inner_mask = distances <= inner_radius
            protein_grid[x_grid[inner_mask], y_grid[inner_mask], z_grid[inner_mask]] -= 100.0
            
            # Outer shell (inner_radius to outer_radius): pocket prediction scores (attraction)
            outer_mask = (distances > inner_radius) & (distances <= outer_radius)
            protein_grid[x_grid[outer_mask], y_grid[outer_mask], z_grid[outer_mask]] += pocket_score
    
    # Create kernels
    kernel1 = create_probe_kernel(probe_radius, grid_spacing, protein_grid.shape)
    kernel2 = create_probe_kernel(5.0, grid_spacing, protein_grid.shape, value=-1.0)
    
    # Calculate scoring grids using FFT convolution
    scoring_grid1 = calculate_score_with_fft(protein_grid, kernel1, probe_radius, grid_spacing, dimensions)
    scoring_grid2 = calculate_score_with_fft(protein_grid, kernel2, 5.0, grid_spacing, dimensions)
        
    # Create coordinate grids for finding points
    x_grid, y_grid, z_grid = np.mgrid[0:dimensions[0], 0:dimensions[1], 0:dimensions[2]]
    grid_points = np.stack([x_grid, y_grid, z_grid], axis=-1) * grid_spacing + min_coords
    
    # Find points with positive scores (attraction zones)
    positive_mask = scoring_grid1 > 1e-4
    positive_points = grid_points[positive_mask]
    positive_scores1 = scoring_grid1[positive_mask]
    positive_scores2 = scoring_grid2[positive_mask]
        
    if len(positive_points) == 0:
        return None
    
    # Sort by score and take top K points
    sorted_indices = np.argsort(positive_scores1)[::-1]  # Descending order
    # top_indices = sorted_indices[:min(top_k_points, len(sorted_indices))]
    top_indices = sorted_indices
    
    top_points = positive_points[top_indices]
    top_scores1 = positive_scores1[top_indices]
    top_scores2 = positive_scores2[top_indices]
    
    clustering = DBSCAN(eps=2.5, min_samples=30).fit(top_points)
    cluster_labels = clustering.labels_
    
    # Get unique clusters (excluding noise which is -1)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]
    
    cluster_centers = []
    center_scores = []  # Store scores for cluster centers
    
    for cluster_id in unique_clusters:
        # Get points in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_points = top_points[cluster_mask]
        cluster_scores1 = top_scores1[cluster_mask]
        cluster_scores2 = top_scores2[cluster_mask]
        
        if len(cluster_points) > 200:  # Large cluster that needs further decomposition
            # Sort points by cluster_scores2 and take top 25%
            sorted_indices = np.argsort(cluster_scores2)[::-1]  # Descending order
            top_half_indices = sorted_indices[:30]
            refined_points = cluster_points[top_half_indices]
            refined_scores1 = cluster_scores1[top_half_indices]
            
            # Perform secondary clustering on refined points
            secondary_clustering = DBSCAN(eps=4, min_samples=1).fit(refined_points)
            secondary_labels = secondary_clustering.labels_
            
            # Process secondary clusters
            secondary_unique_clusters = np.unique(secondary_labels)
            secondary_unique_clusters = secondary_unique_clusters[secondary_unique_clusters != -1]
            
            for secondary_cluster_id in secondary_unique_clusters:
                secondary_cluster_mask = secondary_labels == secondary_cluster_id
                secondary_cluster_points = refined_points[secondary_cluster_mask]
                secondary_cluster_scores = refined_scores1[secondary_cluster_mask]
                
                if len(secondary_cluster_points) > 0:
                    # Use spatial center of the secondary cluster
                    secondary_cluster_center = np.mean(secondary_cluster_points, axis=0)
                    cluster_centers.append(secondary_cluster_center)
                    # Use mean score of points in the cluster
                    center_scores.append(np.mean(secondary_cluster_scores))
        
        else:  # Small cluster, process as before
            if len(cluster_points) > 0:
                # Use spatial center of the cluster
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
                # Use mean score of points in the cluster
                center_scores.append(np.mean(cluster_scores1))
    
    cluster_centers = np.array(cluster_centers)
    center_scores = np.array(center_scores)
    return {
        'centers': cluster_centers,
        'center_scores': center_scores,
        'top_points': top_points,
        'top_scores2': top_scores2,
        'positive_points': positive_points,
        'positive_scores2': positive_scores2
    }

def test_find_pocket_centers():
    """Test the find_pocket_centers_docking_style function using real data from database."""
    print("Starting docking-style pocket centers calculation test...")
    
    print("\nStep 1: Connecting to database...")
    with db_connection() as conn:
        cur = conn.cursor()
        
        print("Step 2: Selecting random entry with pocket predictions...")
        try:
            # First get a random lname that has pocket predictions
            # lname = select_random_with_pocket_pred(cur)
            # lname = '1mky_0'
            lname = '1ish_1'
            # lname = '1qjw_6'
            # lname = '1d1x_0'
            # lname = '1nx8_0'
            # lname = '1nd0_0'
            # lname = '2ay3_2'
            # lname = '1i3n_0'
            print(f"Selected random lname: {lname}")
            
            # Then get the id for this lname
            cur.execute("""
                SELECT m.id
                FROM moad_test_results m
                JOIN processed_datasets p ON m.id = p.id
                WHERE p.lname = %s AND m.pocket_pred IS NOT NULL
            """, (lname,))
            
            result = cur.fetchone()
            if result is None:
                raise ValueError(f"No pocket predictions found for {lname}")
            
            id = result[0]
            pname = lname.split('_')[0]  # Extract protein name
            print(f"Selected entry: {lname} (protein: {pname}, id: {id})")
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        print("Step 3: Fetching protein structure...")
        try:
            pdb_text, struct = get_protein_data(cur, lname)
            print("Protein structure loaded successfully")
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        print("Step 4: Loading pocket predictions from database...")
        try:
            # Get pocket predictions using the helper function
            pocket_pred = get_pocket_pred_data(cur, lname)
            print(f"Pocket predictions loaded: {len(pocket_pred)} residues")
            print(f"Prediction range: {np.min(pocket_pred):.3f} to {np.max(pocket_pred):.3f}")
            print(f"Mean prediction: {np.mean(pocket_pred):.3f}")
            
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        print("\nStep 5: Finding pocket centers using docking-style approach...")
        t0 = time.time()
        results = find_pocket_centers(struct, pocket_pred)
        print(f"Calculation completed in {time.time() - t0:.1f} seconds")
        
        print(f"\nStep 6: Results summary...")
        print(f"Number of pocket centers: {len(results['centers'])}")
        print(f"Number of top scoring points: {len(results['top_points'])}")
        for i, center in enumerate(results['centers']):
            print(f"  Pocket {i+1} center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        
        print("\nStep 7: Saving results to PDB files...")
        
        # Import write_bfactor_to_pdb from clustering module
        from src.clustering import write_bfactor_to_pdb
        
        # Save protein structure with pocket predictions as b-factors
        protein_file = f'protein_with_pockets_{lname}.pdb'
        write_bfactor_to_pdb(pocket_pred, pdb_text, output_file=protein_file, virtual_atom=False)
        print(f"Protein with pocket predictions saved to {protein_file}")
        
        # Save pocket centers with scores1 as b-factors
        centers_file = f'pocket_centers_{lname}.pdb'
        save_points_as_pdb(centers_file, results['centers'], b_factors=results['center_scores'], atom_type="P", residue_name="POC")
        print(f"Pocket centers saved to {centers_file}")
        
        # Save top points with scores2 as b-factors
        top_points_file = f'top_points_{lname}.pdb'
        save_points_as_pdb(top_points_file, results['top_points'], b_factors=results['top_scores2'], atom_type="O", residue_name="TOP")
        print(f"Top points saved to {top_points_file}")
        
        # Save positive points with scores2 as b-factors
        positive_points_file = f'positive_points_{lname}.pdb'
        save_points_as_pdb(positive_points_file, results['positive_points'], b_factors=results['positive_scores2'], atom_type="N", residue_name="POS")
        print(f"Positive points saved to {positive_points_file}")
        
        # Save ligand MOL file
        print("\nStep 8: Saving ligand MOL file...")
        cur.execute("""
            SELECT mol
            FROM ligands
            WHERE name = %s AND protein_name = %s
        """, (lname, pname))
        mol_data = cur.fetchone()
        if mol_data and mol_data[0] is not None:
            mol_file = f'{lname}.mol'
            with open(mol_file, 'wb') as f:
                f.write(mol_data[0].tobytes())
            print(f"Ligand MOL file saved to {mol_file}")
        else:
            print(f"No MOL file found for ligand {lname}")
        
        print("\nTest completed successfully!")

if __name__ == "__main__":
    test_find_pocket_centers()

# %%
