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

def get_solvent_accessible_points(structure: Structure, grid_spacing: float = 0.5, probe_radius: float = 1.5) -> np.ndarray:
    """Generate solvent accessible points around the protein using FFT.
    
    Args:
        structure: Structure object containing protein coordinates
        grid_spacing: Spacing between grid points in Angstroms
        probe_radius: Radius of probe sphere in Angstroms (default 1.4A for water)
        
    Returns:
        np.ndarray: Array of shape (N, 3) containing coordinates of solvent accessible points
    """
    print("\nStep 1: Setting up grid...")
    # Get all protein atoms
    atoms = structure[0].get_atoms()  # Use first model
    coords = np.array([atom.get_coord() for atom in atoms])
    
    # Define grid boundaries with padding
    padding = 10.0  # Add padding around protein
    min_coords = np.min(coords, axis=0) - padding
    max_coords = np.max(coords, axis=0) + padding
    print(f"Grid boundaries: {min_coords} to {max_coords}")
    
    # Calculate grid dimensions
    dimensions = np.ceil((max_coords - min_coords) / grid_spacing).astype(int)
    print(f"Grid dimensions: {dimensions}")
    
    print("Step 2: Creating protein occupancy grid...")
    # Initialize 3D grid
    grid = np.zeros(dimensions, dtype=np.float64)
    
    # Convert atom coordinates to grid indices and place atoms with van der Waals radii
    grid_coords = ((coords - min_coords) / grid_spacing).astype(int)
    
    # Use van der Waals radii for different atoms (simplified)
    vdw_radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    
    for i, (atom, idx) in enumerate(zip(atoms, grid_coords)):
        if np.all(idx >= 0) and np.all(idx < dimensions):
            element = atom.element if atom.element else 'C'  # Use element attribute directly
            vdw_radius = vdw_radii.get(element, 1.7)  # Default to carbon
            
            # Fill grid points within van der Waals radius
            vdw_grid_radius = int(np.ceil(vdw_radius / grid_spacing))
            
            # Create a small grid around the atom - maintain x,y,z ordering
            x_range = slice(max(0, idx[0] - vdw_grid_radius), min(dimensions[0], idx[0] + vdw_grid_radius + 1))
            y_range = slice(max(0, idx[1] - vdw_grid_radius), min(dimensions[1], idx[1] + vdw_grid_radius + 1))
            z_range = slice(max(0, idx[2] - vdw_grid_radius), min(dimensions[2], idx[2] + vdw_grid_radius + 1))
            
            # Get grid coordinates for this region - consistent with x,y,z ordering
            x_grid, y_grid, z_grid = np.mgrid[x_range, y_range, z_range]
            
            # Calculate distances in grid units
            distances = np.sqrt((x_grid - idx[0])**2 + (y_grid - idx[1])**2 + (z_grid - idx[2])**2) * grid_spacing
            
            # Mark grid points within van der Waals radius as occupied
            occupied_mask = distances <= vdw_radius
            grid[x_grid[occupied_mask], y_grid[occupied_mask], z_grid[occupied_mask]] = 1.0
            
    print("Step 3: Creating probe kernel...")
    # Create probe kernel (water probe)
    probe_radius_grid = int(np.ceil(probe_radius / grid_spacing))
    kernel_size = 2 * probe_radius_grid + 1
    
    # Use consistent x,y,z ordering (note: mgrid returns in the order specified)
    x, y, z = np.mgrid[-probe_radius_grid:probe_radius_grid + 1,
                       -probe_radius_grid:probe_radius_grid + 1,
                       -probe_radius_grid:probe_radius_grid + 1]
    
    distances = np.sqrt(x**2 + y**2 + z**2) * grid_spacing
    kernel = (distances <= probe_radius).astype(np.float64)
    
    print("Step 4: Performing FFT convolution...")
    t0 = time.time()
    # Pad kernel to match grid size for FFT convolution
    kernel_padded = np.zeros(grid.shape, dtype=np.float64)
    kernel_padded[:kernel.shape[0], :kernel.shape[1], :kernel.shape[2]] = kernel
    
    # Perform FFT convolution
    grid_fft = np.array(fftn(grid))
    kernel_fft = np.array(fftn(kernel_padded))
    
    # Element-wise multiplication in frequency domain
    convolution_fft = grid_fft * kernel_fft
    
    # Transform back to spatial domain
    convolution = np.real(np.array(ifftn(convolution_fft)))
    print(f"FFT convolution completed in {time.time() - t0:.1f} seconds")
    
    print("Step 5: Finding solvent accessible points...")
    # Points are solvent accessible if probe doesn't overlap with protein
    # This means convolution value should be zero (no protein atoms within probe radius)
    solvent_mask = convolution < 1e-10
    
    # Convert back to coordinates - np.where returns indices in array order (x,y,z for our grid)
    x_indices, y_indices, z_indices = np.where(solvent_mask)
    solvent_coords = np.stack([x_indices, y_indices, z_indices], axis=1).astype(np.float64)
    solvent_coords = solvent_coords * grid_spacing + min_coords
    
    print("Step 6: Filtering distant points...")
    # Filter points that are too far from the protein surface
    max_distance = 15.0  # Maximum distance from protein surface
    
    # Create a distance map from protein surface
    protein_surface = (convolution > 0) & (convolution < np.sum(kernel))  # Partial overlap
    if np.any(protein_surface):
        distance_map = np.array(ndimage.distance_transform_edt(~protein_surface)) * grid_spacing
        close_points_mask = distance_map[solvent_mask] <= max_distance
        solvent_coords = solvent_coords[close_points_mask]
    
    print(f"Found {len(solvent_coords)} solvent accessible points")
    return solvent_coords

def get_bounding_box(coords: np.ndarray, padding: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Get bounding box for a set of coordinates with padding.
    
    Args:
        coords: Array of shape (N, 3) containing coordinates
        padding: Padding to add around the box in Angstroms
        
    Returns:
        Tuple of (min_coords, max_coords) defining the box
    """
    min_coords = np.min(coords, axis=0) - padding
    max_coords = np.max(coords, axis=0) + padding
    return min_coords, max_coords

def points_in_box(points: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray) -> np.ndarray:
    """Get points that lie within a bounding box.
    
    Args:
        points: Array of shape (N, 3) containing coordinates
        min_coords: Minimum coordinates of box
        max_coords: Maximum coordinates of box
        
    Returns:
        Boolean mask indicating which points are in the box
    """
    return np.all((points >= min_coords) & (points <= max_coords), axis=1)

def calculate_cluster_centers(structure: Structure, clusters: np.ndarray) -> np.ndarray:
    """Calculate center positions for each cluster.
    
    Args:
        structure: Structure object containing protein coordinates
        clusters: Array of cluster assignments for each residue
        grid_spacing: Spacing between grid points in Angstroms
        
    Returns:
        np.ndarray: Array of shape (n_clusters, 3) containing center coordinates for each cluster
    """
    # Get solvent accessible points
    solvent_points = get_solvent_accessible_points(structure)
    
    # Get residue coordinates (all atoms)
    residues = structure[0].get_residues()  # Use first model
    
    # Get unique clusters (excluding noise which is -1)
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters != -1]
    
    # Calculate center for each cluster
    centers = []
    for cluster_id in unique_clusters:
        # Get coordinates of all atoms in residues that belong to this cluster
        cluster_mask = clusters == cluster_id
        cluster_atom_coords = []
        
        for res_idx, residue in enumerate(residues):
            if res_idx < len(cluster_mask) and cluster_mask[res_idx]:
                # Get all atoms in this residue
                for atom in residue.get_atoms():
                    cluster_atom_coords.append(atom.get_coord())
        
        if len(cluster_atom_coords) == 0:
            continue
            
        cluster_atom_coords = np.array(cluster_atom_coords)
        
        # Get bounding box for this cluster
        min_coords, max_coords = get_bounding_box(cluster_atom_coords)
        
        # Get solvent points within the box
        box_mask = points_in_box(solvent_points, min_coords, max_coords)
        local_solvent_points = solvent_points[box_mask]
        
        if len(local_solvent_points) == 0:
            # If no solvent points in box, use geometric center of cluster
            centers.append(np.mean(cluster_atom_coords, axis=0))
            continue
        
        # Calculate distances from each solvent point to all atoms in the cluster
        distances = np.linalg.norm(
            local_solvent_points[:, np.newaxis] - cluster_atom_coords, 
            axis=2
        )
        # Find point with minimum sum of squared distances to all atoms in cluster
        sum_of_squares = np.sum(distances**2, axis=1)
        best_idx = np.argmin(sum_of_squares)
        centers.append(local_solvent_points[best_idx])
    
    return np.array(centers)

def save_points_as_pdb(output_file: str, points: np.ndarray, atom_type: str = "O", residue_name: str = "SOL"):
    """Save points to a PDB file as HETATM records.
    
    Args:
        output_file: Path to output PDB file
        points: Array of shape (N, 3) containing point coordinates
        atom_type: Atom type for the points (default: "O")
        residue_name: Residue name for the points (default: "SOL")
    """
    # Randomly sample 1000 points if there are more
    if len(points) > 1000:
        indices = np.random.choice(len(points), 1000, replace=False)
        sampled_points = points[indices]
    else:
        sampled_points = points
    
    with open(output_file, 'w') as f:
        # Write points as HETATM records
        for i, point in enumerate(sampled_points, 1):
            hetatm = (
                f"HETATM{i:5d}  {atom_type}   {residue_name} X{i:4d}    "
                f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}"
                f"  1.00  0.00           {atom_type}  \n"
            )
            f.write(hetatm)
        
        f.write("END\n")

def select_random(cur):
    """Select a random lname with cluster data from database.
    
    Args:
        cur: Database cursor
        
    Returns:
        str: lname
    """
    cur.execute("""
        SELECT p.lname
        FROM moad_test_results m
        JOIN processed_datasets p ON m.id = p.id
        WHERE m.clusters IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 1
    """)
    
    result = cur.fetchone()
    if result is None:
        raise ValueError("No entries found with cluster data!")
    
    return result[0]

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

def get_cluster_data(cur, lname):
    """Get cluster data for a specific lname.
    
    Args:
        cur: Database cursor
        lname: Ligand name
        
    Returns:
        numpy.ndarray: clusters data
    """
    cur.execute("""
        SELECT m.clusters
        FROM moad_test_results m
        JOIN processed_datasets p ON m.id = p.id
        WHERE p.lname = %s
    """, (lname,))
    
    result = cur.fetchone()
    if result is None:
        raise ValueError(f"No cluster data found for {lname}")
    
    return pickle.loads(result[0])

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

def get_true_pocket_data(cur, lname):
    """Get true pocket data for a specific lname.
    
    Args:
        cur: Database cursor
        lname: Ligand name
        
    Returns:
        numpy.ndarray: true pocket data (boolean array)
    """
    cur.execute("""
        SELECT is_pocket
        FROM processed_datasets
        WHERE lname = %s
    """, (lname,))
    
    result = cur.fetchone()
    if result is None:
        raise ValueError(f"No true pocket data found for {lname}")
    
    return pickle.loads(result[0])

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

def test_calculate_cluster_centers():
    """Test the calculate_cluster_centers function using real cluster data from database."""
    print("Starting cluster centers calculation test...")
    
    print("\nStep 1: Connecting to database...")
    with db_connection() as conn:
        cur = conn.cursor()
        
        print("Step 2: Selecting random entry with cluster data...")
        try:
            # lname = select_random(cur)
            lname = '1d1x_0'
            pname = lname.split('_')[0]  # Extract protein name
            print(f"Selected entry: {lname} (protein: {pname})")
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
        
        print("Step 4: Loading cluster data...")
        try:
            clusters = get_cluster_data(cur, lname)
            print(f"Cluster data loaded: {len(clusters)} residues")
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # Get unique clusters (excluding noise which is -1)
        unique_clusters = np.unique(clusters)
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        print(f"Found {len(unique_clusters)} clusters:")
        for cluster_id in unique_clusters:
            count = np.sum(clusters == cluster_id)
            print(f"  Cluster {cluster_id}: {count} residues")
        
        noise_count = np.sum(clusters == -1)
        print(f"  Noise points: {noise_count} residues")
        
        print("\nStep 5: Calculating cluster centers...")
        t0 = time.time()
        cluster_centers = calculate_cluster_centers(struct, clusters)
        print(f"Calculation completed in {time.time() - t0:.1f} seconds")
        
        print(f"\nStep 6: Results summary...")
        print(f"Number of cluster centers: {len(cluster_centers)}")
        for i, center in enumerate(cluster_centers):
            print(f"  Cluster {unique_clusters[i]} center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        
        print("\nStep 7: Saving results to PDB files...")
        
        # Import write_bfactor_to_pdb from clustering module
        from src.clustering import write_bfactor_to_pdb
        
        # Save protein structure with cluster assignments as b-factors
        protein_file = f'protein_with_clusters_{lname}.pdb'
        # Convert cluster assignments to B-factors (cluster ID + 1, so 0 becomes 1, -1 becomes 0)
        bfactors = np.where(clusters == -1, 0, clusters + 1)
        write_bfactor_to_pdb(bfactors, pdb_text, output_file=protein_file, virtual_atom=False)
        print(f"Protein with cluster assignments saved to {protein_file}")
        
        # Save cluster centers
        centers_file = f'cluster_centers_{lname}.pdb'
        save_points_as_pdb(centers_file, cluster_centers, atom_type="C", residue_name="CEN")
        print(f"Cluster centers saved to {centers_file}")
        
        print("\nTest completed successfully!")



def test_solvent_accessible_points():
    """Test the solvent accessible points calculation using a random protein from database."""
    print("Starting solvent accessible points test...")
    
    print("\nStep 1: Connecting to database...")
    with db_connection() as conn:
        cur = conn.cursor()
        
        print("Step 2: Selecting random protein...")
        # First get a random protein name
        cur.execute("""
            SELECT name 
            FROM proteins 
            WHERE pdb IS NOT NULL 
            ORDER BY RANDOM() 
            LIMIT 1
        """)
        pdb_id = cur.fetchone()[0]
        
        print(f"Selected protein: {pdb_id}")
        print("Step 3: Fetching protein data...")
        # Then get the PDB data for this protein
        cur.execute("""
            SELECT pdb 
            FROM proteins 
            WHERE name = %s
        """, (pdb_id,))
        pdb_data = cur.fetchone()[0]
        pdb_text = pdb_data.tobytes().decode('utf-8')
        
        print("Step 4: Creating structure object...")
        # Create structure object
        struct = Structure()
        struct.read(io.StringIO(pdb_text))
        
        print("\nStep 5: Calculating solvent accessible points...")
        t0 = time.time()
        solvent_points = get_solvent_accessible_points(struct)
        print(f"Calculation completed in {time.time() - t0:.1f} seconds")
        
        print("\nStep 6: Saving results...")
        # Save results to PDB file
        output_file = f'solvent_points_{pdb_id}.pdb'
        save_points_as_pdb(output_file, solvent_points, atom_type="O", residue_name="SOL")
        print(f"Results saved to {output_file}")
        
        print("\nStep 7: Calculating statistics...")
        atoms = struct[0].get_atoms()
        protein_coords = np.array([atom.get_coord() for atom in atoms])
        print(f"Number of protein atoms: {len(protein_coords)}")
        print(f"Number of solvent points: {len(solvent_points)}")
        
        print("Calculating distances (sampling 1000 points)...")
        min_distances = []
        for point in solvent_points[:1000]:  # Sample first 1000 points for speed
            distances = np.linalg.norm(protein_coords - point, axis=1)
            min_distances.append(np.min(distances))
        
        min_distances = np.array(min_distances)
        print(f"Average minimum distance to protein: {np.mean(min_distances):.2f}Å")
        print(f"Min distance to protein: {np.min(min_distances):.2f}Å")
        print(f"Max distance to protein: {np.max(min_distances):.2f}Å")
        
        print("\nTest completed successfully!")

def find_pocket_centers(structure: Structure, pocket_pred: np.ndarray, prob_threshold: float = 0.07,
                                     grid_spacing: float = 0.5, probe_radius: float = 2.5,
                                     inner_radius: float = 1.5, outer_radius: float = 2,
                                     top_k_points: int = 8000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find pocket centers using FFT-based docking-like scoring approach.
    
    This function creates a scoring grid around the protein using FFT convolution:
    - Inner shell (within inner_radius): negative values (repulsion)
    - Middle shell (inner_radius to outer_radius): pocket prediction scores (attraction)
    - Outer shell (outer_radius to probe_radius): pocket prediction scores (attraction)
    - Beyond probe_radius: zero
    
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
        np.ndarray: Array of shape (n_clusters, 3) containing pocket center coordinates
    """
    # Get all protein atoms
    atoms = structure[0].get_atoms()
    coords = np.array([atom.get_coord() for atom in atoms])
    
    # Get residue information for pocket predictions
    residues = structure[0].get_residues()
    residue_list = list(residues)
    
    # Create mapping from atom to residue index
    atom_to_residue = []
    for atom in atoms:
        # Find which residue this atom belongs to
        for res_idx, residue in enumerate(residue_list):
            if atom in residue.atoms:
                atom_to_residue.append(res_idx)
                break
        else:
            # If not found, use 0 as default
            atom_to_residue.append(0)
    
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
    
    # Create a simple probe kernel with probe_radius
    # This kernel will be used to find points that are close to the protein surface
    probe_radius_grid = int(np.ceil(probe_radius / grid_spacing))
    
    x, y, z = np.mgrid[-probe_radius_grid:probe_radius_grid + 1,
                       -probe_radius_grid:probe_radius_grid + 1,
                       -probe_radius_grid:probe_radius_grid + 1]
    
    distances = np.sqrt(x**2 + y**2 + z**2) * grid_spacing
    
    # Create simple probe kernel (all positive values within probe_radius)
    kernel = np.where(distances <= probe_radius, 1.0, 0.0).astype(np.float64)
    
    # Pad kernel to match grid size for FFT convolution
    kernel_padded = np.zeros(protein_grid.shape, dtype=np.float64)
    kernel_padded[:kernel.shape[0], :kernel.shape[1], :kernel.shape[2]] = kernel
    
    # Perform FFT convolution
    protein_fft = np.array(fftn(protein_grid))
    kernel_fft = np.array(fftn(kernel_padded))
    
    # Element-wise multiplication in frequency domain
    convolution_fft = protein_fft * kernel_fft
    
    # Transform back to spatial domain
    scoring_grid_ = np.real(np.array(ifftn(convolution_fft)))
    scoring_grid = np.zeros(protein_grid.shape, dtype=np.float64)
    scoring_grid[:dimensions[0]-probe_radius_grid, :dimensions[1]-probe_radius_grid, :dimensions[2]-probe_radius_grid] = scoring_grid_[probe_radius_grid:, probe_radius_grid:, probe_radius_grid:]
    
    # Create coordinate grids for finding points
    x_grid, y_grid, z_grid = np.mgrid[0:dimensions[0], 0:dimensions[1], 0:dimensions[2]]
    grid_points = np.stack([x_grid, y_grid, z_grid], axis=-1) * grid_spacing + min_coords
    
    # Find points with positive scores (attraction zones)
    positive_mask = scoring_grid > 1e-4
    positive_points = grid_points[positive_mask]
    positive_scores = scoring_grid[positive_mask]
    
    if len(positive_points) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Sort by score and take top K points
    sorted_indices = np.argsort(positive_scores)[::-1]  # Descending order
    top_indices = sorted_indices[:min(top_k_points, len(sorted_indices))]
    
    top_points = positive_points[top_indices]
    top_scores = positive_scores[top_indices]
    
    clustering = DBSCAN(eps=2.5, min_samples=30).fit(top_points)
    cluster_labels = clustering.labels_
    
    # Get unique clusters (excluding noise which is -1)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]
    
    cluster_centers = []
    
    for cluster_id in unique_clusters:
        # Get points in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_points = top_points[cluster_mask]
        cluster_scores = top_scores[cluster_mask]
        
        if len(cluster_points) > 0:
            # Use spatial center of the cluster instead of highest scoring point
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
    
    cluster_centers = np.array(cluster_centers)
    return cluster_centers, top_points, positive_points

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
            lname = '1d1x_0'
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
        pocket_centers, top_points, positive_points = find_pocket_centers(struct, pocket_pred)
        print(f"Calculation completed in {time.time() - t0:.1f} seconds")
        
        print(f"\nStep 6: Results summary...")
        print(f"Number of pocket centers: {len(pocket_centers)}")
        print(f"Number of top scoring points: {len(top_points)}")
        for i, center in enumerate(pocket_centers):
            print(f"  Pocket {i+1} center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        
        print("\nStep 7: Saving results to PDB files...")
        
        # Import write_bfactor_to_pdb from clustering module
        from src.clustering import write_bfactor_to_pdb
        
        # Save protein structure with pocket predictions as b-factors
        protein_file = f'protein_with_pockets_{lname}.pdb'
        write_bfactor_to_pdb(pocket_pred, pdb_text, output_file=protein_file, virtual_atom=False)
        print(f"Protein with pocket predictions saved to {protein_file}")
        
        # Save pocket centers
        centers_file = f'pocket_centers_{lname}.pdb'
        save_points_as_pdb(centers_file, pocket_centers, atom_type="P", residue_name="POC")
        print(f"Pocket centers saved to {centers_file}")
        
        # Save top points
        top_points_file = f'top_points_{lname}.pdb'
        save_points_as_pdb(top_points_file, top_points, atom_type="O", residue_name="TOP")
        print(f"Top points saved to {top_points_file}")
        
        # Save positive points
        positive_points_file = f'positive_points_{lname}.pdb'
        save_points_as_pdb(positive_points_file, positive_points, atom_type="N", residue_name="POS")
        print(f"Positive points saved to {positive_points_file}")
        
        print("\nTest completed successfully!")

if __name__ == "__main__":
    # test_solvent_accessible_points()
    # test_calculate_cluster_centers()
    
    # Test with a random ligand
    test_find_pocket_centers()

# %%
