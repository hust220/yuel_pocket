
import os
import sys
import random
import numpy as np
from io import StringIO
from typing import Tuple, List, Optional
import time
from scipy.fft import fftn, ifftn
from scipy import ndimage

# Add project root to path
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)

# Only import necessary modules that don't cause circular imports or missing dependency errors
from src.residues.dataset import PocketDataset
from src.pdb_utils import Structure, Atom

# --- Define functions from pocket_utils2.py locally to avoid import issues ---

def get_bounding_box(coords: np.ndarray, padding: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Get bounding box for a set of coordinates with padding."""
    min_coords = np.min(coords, axis=0) - padding
    max_coords = np.max(coords, axis=0) + padding
    return min_coords, max_coords

def points_in_box(points: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray) -> np.ndarray:
    """Get points that lie within a bounding box."""
    return np.all((points >= min_coords) & (points <= max_coords), axis=1)

def get_solvent_accessible_points(structure: Structure, grid_spacing: float = 1.5, probe_radius: float = 1.5) -> np.ndarray:
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
    
    # Convert atom coordinates to grid indices
    grid_coords = ((coords - min_coords) / grid_spacing).astype(int)
    
    # Use van der Waals radii for different atoms (simplified)
    vdw_radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    
    for i, (atom, idx) in enumerate(zip(atoms, grid_coords)):
        if np.all(idx >= 0) and np.all(idx < dimensions):
            element = atom.element if atom.element else 'C'  # Use element attribute directly
            vdw_radius = vdw_radii.get(element, 1.7)  # Default to carbon
            
            # Fill grid points within van der Waals radius AND shell
            # Shell thickness assumed to be probe_radius
            search_radius = vdw_radius + probe_radius
            search_grid_radius = int(np.ceil(search_radius / grid_spacing))
            
            # Create a small grid around the atom - maintain x,y,z ordering
            # Use search_grid_radius instead of vdw_grid_radius
            x_range = slice(max(0, idx[0] - search_grid_radius), min(dimensions[0], idx[0] + search_grid_radius + 1))
            y_range = slice(max(0, idx[1] - search_grid_radius), min(dimensions[1], idx[1] + search_grid_radius + 1))
            z_range = slice(max(0, idx[2] - search_grid_radius), min(dimensions[2], idx[2] + search_grid_radius + 1))
            
            # Get grid coordinates for this region - consistent with x,y,z ordering
            x_grid, y_grid, z_grid = np.mgrid[x_range, y_range, z_range]
            
            # Calculate distances in grid units
            distances = np.sqrt((x_grid - idx[0])**2 + (y_grid - idx[1])**2 + (z_grid - idx[2])**2) * grid_spacing
            
            # Mark grid points 
            # Occupied mask: inside VDW radius -> -100
            occupied_mask = distances <= vdw_radius
            # Shell mask: just outside VDW radius -> 1
            shell_mask = (distances > vdw_radius) & (distances <= search_radius)
            
            # Get current grid values to avoid overwriting existing core values
            current_grid_vals = grid[x_grid, y_grid, z_grid]
            
            # Update values
            # Only calculate updates where necessary to avoid modifying array in place incorrectly
            # We want to write -100 where occupied
            # We want to write 1 where shell AND NOT already -100
            
            # Use where to construct the new block values
            # If occupied -> -100
            # Else if shell AND current != -100 -> 1
            # Else -> current
            
            new_vals = np.where(occupied_mask, -100.0, 
                               np.where(shell_mask & (current_grid_vals != -100.0), 1.0, current_grid_vals))
            
            grid[x_grid, y_grid, z_grid] = new_vals
            
    print("Step 3: Creating probe kernel...")
    # Create probe kernel (water probe)
    probe_radius_grid = int(np.ceil(probe_radius / grid_spacing))
    # kernel_size = 2 * probe_radius_grid + 1 # Unused variable
    
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
    shell_mask = convolution > 1e-10
    
    # Convert back to coordinates - np.where returns indices in array order (x,y,z for our grid)
    x_indices, y_indices, z_indices = np.where(shell_mask)
    shell_coords = np.stack([x_indices, y_indices, z_indices], axis=1).astype(np.float64)
    shell_coords = shell_coords * grid_spacing + min_coords
    
    # Get scores
    scores = convolution[shell_mask]
    
    print(f"Found {len(shell_coords)} solvent accessible points")
    return shell_coords, scores

def save_points_as_pdb(output_file: str, points: np.ndarray, atom_type: str = "O", residue_name: str = "SOL"):
    """Save points to a PDB file as HETATM records.
    
    Args:
        output_file: Path to output PDB file
        points: Array of shape (N, 3) containing point coordinates
        atom_type: Atom type for the points (default: "O")
        residue_name: Residue name for the points (default: "SOL")
    """
    # No random sampling, just save all provided points
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

from scipy.spatial import KDTree
import argparse

# --- Shrake-Rupley Algorithm Implementation ---

def generate_sphere_points(n: int) -> np.ndarray:
    """Generate n uniformly distributed points on a unit sphere using Fibonacci lattice."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.stack([x, y, z], axis=1)

def get_sas_points_shrake_rupley(structure: Structure, probe_radius: float = 1.4, n_points_per_atom: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """Generate solvent accessible surface points using Shrake-Rupley algorithm.
    
    Args:
        structure: Structure object
        probe_radius: Probe radius (default 1.4A for water)
        n_points_per_atom: Number of points to sample per atom
        
    Returns:
        tuple: (points, scores) 
               Note: 'scores' here will be dummy values (e.g. 1.0) as SR is geometric, 
               or could be potential/depth if calculated. We'll return 1.0s.
    """
    print(f"\nMethod: Shrake-Rupley (n={n_points_per_atom}, probe={probe_radius})")
    
    atoms = structure[0].get_atoms()
    coords = np.array([atom.get_coord() for atom in atoms])
    n_atoms = len(atoms)
    
    # Define VDW radii
    vdw_radii_dict = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'H': 1.2}
    radii = []
    for atom in atoms:
        element = atom.element if atom.element else 'C'
        radii.append(vdw_radii_dict.get(element, 1.7))
    radii = np.array(radii)
    
    # Expanded radii for SAS check
    expanded_radii = radii + probe_radius
    
    # Build neighbor search tree
    tree = KDTree(coords)
    
    sas_points = []
    
    # Pre-generate unit sphere points
    unit_sphere = generate_sphere_points(n_points_per_atom)
    
    print("Generating surface points...")
    # This loop can be slow in pure Python for large proteins, but is standard for SR
    for i in range(n_atoms):
        center = coords[i]
        r = expanded_radii[i]
        
        # Generate points for this atom
        atom_sphere_points = center + unit_sphere * r
        
        # Find neighbors that could overlap
        # Max possible overlap distance is r + max_other_r. 
        # A conservative upper bound is sufficient.
        max_possible_r = np.max(expanded_radii)
        neighbor_indices = tree.query_ball_point(center, r + max_possible_r)
        
        # Filter neighbors: exclude self
        neighbor_indices = [idx for idx in neighbor_indices if idx != i]
        
        if not neighbor_indices:
            sas_points.append(atom_sphere_points)
            continue
            
        neighbor_coords = coords[neighbor_indices]
        neighbor_radii = expanded_radii[neighbor_indices]
        
        # Vectorized check for point occlusion
        # A point is blocked if dist(point, neighbor) < neighbor_radius
        
        # Shape: (n_points, n_neighbors)
        # We need to process in chunks if neighbors are too many, but usually manageable
        
        # Using broadcasting:
        # P: (n_points, 1, 3)
        # N: (1, n_neighbors, 3)
        # DistSq: (n_points, n_neighbors)
        
        P = atom_sphere_points[:, np.newaxis, :]
        N = neighbor_coords[np.newaxis, :, :]
        R_sq = neighbor_radii**2
        
        dists_sq = np.sum((P - N)**2, axis=2)
        
        # occlusion_mask[j] is True if point j is occluded by ANY neighbor
        # blocked if dist_sq < neighbor_radius_sq - epsilon
        is_blocked = np.any(dists_sq < R_sq[np.newaxis, :] - 1e-6, axis=1)
        
        valid_points = atom_sphere_points[~is_blocked]
        if len(valid_points) > 0:
            sas_points.append(valid_points)
            
    if not sas_points:
        return np.array([]), np.array([])
        
    all_points = np.vstack(sas_points)
    # Dummy scores (1.0) since this method is binary (surface vs hidden)
    scores = np.ones(len(all_points))
    
    return all_points, scores

# --- End function definitions ---

def test_random_protein_solvent_points(method='fft'):
    print("Initializing PocketDataset (train split)...")
    # Initialize dataset to get access to Plinder data
    # Dataset init might take a moment to load parquet.
    try:
        dataset = PocketDataset(split='train')
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Pick a random system
    random_idx = random.randint(0, len(dataset) - 1)
    system_id = dataset.ids[random_idx]
    print(f"Selected random system: {system_id}")
    
    # Retrieve raw data using internal method to get PDB string
    # _read_from_zip returns (system_id, receptor_pdb, ligand_mol)
    try:
        _, receptor_pdb_content, _ = dataset._read_from_zip(system_id)
    except Exception as e:
        print(f"Failed to read data for {system_id}: {e}")
        return

    # Parse into Structure object
    print("Parsing protein structure...")
    structure = Structure()
    structure.read(StringIO(receptor_pdb_content))
    
    # Find solvent accessible points
    print(f"Calculating solvent accessible points using {method.upper()} method...")
    
    try:
        if method == 'fft':
            # FFT parameters
            solvent_points, scores = get_solvent_accessible_points(structure, grid_spacing=2, probe_radius=1.5)
        elif method == 'sr':
            # Shrake-Rupley parameters
            solvent_points, scores = get_sas_points_shrake_rupley(structure, probe_radius=1.5, n_points_per_atom=15)
        else:
            print(f"Unknown method: {method}")
            return
            
        num_points = len(solvent_points)
        print(f"Found {num_points} solvent accessible points.")
        
        avg_score = np.mean(scores) if num_points > 0 else 0
        print(f"Total points: {num_points}")
        print(f"Overall average score: {avg_score:.4f}")
        
    except Exception as e:
        print(f"Error calculating solvent points: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Select top points (if FFT) or random sample (if SR, since scores are uniform)
    target_count = 10000
    if num_points > 0:
        if method == 'fft':
            # Sort indices by score descending for FFT
            sorted_indices = np.argsort(scores)[::-1]
            top_indices = sorted_indices[:target_count]
            top_points = solvent_points[top_indices]
            top_scores = scores[top_indices]
            avg_top_score = np.mean(top_scores)
            print(f"Top {len(top_points)} points average convolution score: {avg_top_score:.4f}")
        else:
            # Random sample for SR
            if num_points > target_count:
                indices = np.random.choice(num_points, target_count, replace=False)
                top_points = solvent_points[indices]
            else:
                top_points = solvent_points
            print(f"Sampled {len(top_points)} points from surface.")
    else:
        top_points = np.array([])
        print("No points found!")
    
    # Define output filenames
    output_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_filename = os.path.join(output_dir, f"{system_id}_protein.pdb")
    points_filename = os.path.join(output_dir, f"{system_id}_solvent_points_{method}.pdb")
    
    # Save Protein PDB
    print(f"Saving protein structure to {pdb_filename}...")
    try:
        with open(pdb_filename, 'w') as f:
            f.write(receptor_pdb_content)
    except Exception as e:
        print(f"Error saving protein PDB: {e}")
        
    # Save Solvent Points
    print(f"Saving solvent points to {points_filename}...")
    try:
        save_points_as_pdb(points_filename, top_points, atom_type="O", residue_name="SOL")
    except Exception as e:
        print(f"Error saving solvent points PDB: {e}")
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate solvent accessible points.')
    parser.add_argument('--method', type=str, default='fft', choices=['fft', 'sr'], 
                        help='Method to use: fft (Grid Convolution) or sr (Shrake-Rupley)')
    args = parser.parse_args()
    
    test_random_protein_solvent_points(method=args.method)

