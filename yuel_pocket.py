#%%

import argparse
import os
import numpy as np
import torch
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO, Select
import psycopg2
import tempfile
import io

from src import const
from src.datasets import parse_molecule, get_edges
from src.lightning import YuelPocket
from src.utils import set_deterministic
from src.db_utils import db_connection

def aa_one_hot(residue):
    """Convert amino acid residue to one-hot encoding."""
    n = const.N_RESIDUE_TYPES
    one_hot = np.zeros(n)
    if residue not in const.RESIDUE2IDX:
        residue = 'UNK'
    one_hot[const.RESIDUE2IDX[residue]] = 1
    return one_hot

def parse_protein_structure(struct):
    """Parse protein structure to get positions and one-hot encodings.
    
    Returns:
        protein_pos: shape (n_residues, 3)
        protein_one_hot: shape (n_residues, n_residue_types)
        residue_ids: shape (n_atoms,)
        atom_coords: shape (n_atoms, 3)
        protein_backbone: shape (n_backbone_edges, 2) where each row is [res_idx, res_idx+1]
    """
    protein_pos = []  # Will be shape: (n_residues, 3)
    protein_one_hot = []  # Will be shape: (n_residues, n_residue_types)
    residue_ids = []  # Will be shape: (n_atoms,)
    atom_coords = []  # Will be shape: (n_atoms, 3)
    protein_backbone = []  # Will be shape: (n_backbone_edges, 2)
    
    prev_residue = None
    prev_residue_idx = -1

    for chain in struct.get_chains():
        for residue in chain.get_residues():
            residue_name = residue.get_resname()
            atom_names = [atom.get_name() for atom in residue.get_atoms()]
            if 'CA' in atom_names:
                ca = next(atom for atom in residue.get_atoms() if atom.get_name() == 'CA')
                atom_coords.extend([atom.get_coord() for atom in residue.get_atoms()])
                residue_ids.extend([len(protein_pos)] * len(atom_names))
                
                pos = ca.get_coord().tolist()
                protein_pos.append(pos)
                protein_one_hot.append(aa_one_hot(residue_name))

                # Check if this residue is continuous with the previous one
                if prev_residue is not None:
                    # Get residue numbers
                    prev_num = prev_residue.get_id()[1]
                    curr_num = residue.get_id()[1]
                    
                    # Check if residues are in the same chain and consecutive
                    if (prev_residue.get_parent() == residue.get_parent() and  # Same chain
                        curr_num == prev_num + 1):  # Consecutive numbers
                        protein_backbone.append([prev_residue_idx, len(protein_pos) - 1])

                prev_residue = residue
                prev_residue_idx = len(protein_pos) - 1

    return (np.array(protein_pos), np.array(protein_one_hot), 
            np.array(residue_ids), np.array(atom_coords),
            np.array(protein_backbone))

def calculate_protein_contacts(protein_pos, contact_threshold=10.0, box_size=10.0):
    """Calculate protein contacts using grid-based approach.
    
    Args:
        protein_pos: shape (n_residues, 3)
    Returns:
        protein_contacts: shape (n_contacts, 3) where each row is [res1_idx, res2_idx, distance]
    """
    protein_contacts = []
    
    # Set up grid
    grid = {}
    range_min = [float('inf')] * 3
    range_max = [float('-inf')] * 3

    # Determine range of coordinates
    for pos in protein_pos:
        for j in range(3):
            range_min[j] = min(range_min[j], pos[j])
            range_max[j] = max(range_max[j], pos[j])

    # Assign CA atoms to grid cells
    for i, pos in enumerate(protein_pos):
        id_x = int((pos[0] - range_min[0]) / box_size)
        id_y = int((pos[1] - range_min[1]) / box_size)
        id_z = int((pos[2] - range_min[2]) / box_size)
        grid_key = (id_x, id_y, id_z)
        if grid_key not in grid:
            grid[grid_key] = []
        grid[grid_key].append(i)

    # Find contacts using grid
    for i, pos1 in enumerate(protein_pos):
        id_x = int((pos1[0] - range_min[0]) / box_size)
        id_y = int((pos1[1] - range_min[1]) / box_size)
        id_z = int((pos1[2] - range_min[2]) / box_size)

        # Check neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (id_x + dx, id_y + dy, id_z + dz)
                    if neighbor_key in grid:
                        for j in grid[neighbor_key]:
                            if j > i:  # Avoid duplicate pairs
                                pos2 = protein_pos[j]
                                dist = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))
                                if dist < contact_threshold:
                                    protein_contacts.append([i + 1, j + 1, dist])

    return np.array(protein_contacts)

def prepare_model_input(protein_pos, protein_one_hot, protein_contacts, protein_backbone, 
                       mol_pos, mol_one_hot, mol_bonds, device):
    """Prepare input data for the model.
    
    Args:
        protein_pos: shape (n_residues, 3)
        protein_one_hot: shape (n_residues, n_residue_types)
        protein_contacts: shape (n_contacts, 3)
        protein_backbone: shape (n_residues-1, 2)
        mol_pos: shape (n_atoms, 3)
        mol_one_hot: shape (n_atoms, n_atom_types)
        mol_bonds: shape (n_bonds, 2+bond_features)
        device: torch device
    
    Returns:
        dict containing tensors:
            one_hot: shape (1, n_residues+1+n_atoms, n_residue_types+1+n_atom_types)
            edge_index: shape (1, n_edges, 2)
            edge_attr: shape (1, n_edges, n_edge_features)
            node_mask: shape (1, n_residues+1+n_atoms, 1)
            edge_mask: shape (1, n_edges, 1)
            protein_mask: shape (1, n_residues+1+n_atoms, 1)
    """
    protein_size = len(protein_pos)  # n_residues
    mol_size = len(mol_pos)  # n_atoms

    # Expand one-hot encodings
    protein_one_hot = np.array(protein_one_hot)  # (n_residues, n_residue_types)
    mol_one_hot = np.array(mol_one_hot)  # (n_atoms, n_atom_types)
    
    protein_node_features = protein_one_hot.shape[1]  # n_residue_types
    mol_node_features = mol_one_hot.shape[1]  # n_atom_types

    # Concatenate one-hot encodings with padding
    protein_one_hot = np.concatenate([protein_one_hot, np.zeros((protein_size, mol_node_features+1))], axis=-1)  # (n_residues, n_residue_types+n_atom_types+1)
    joint_one_hot = np.concatenate([np.zeros((1, protein_node_features)), np.ones((1, 1)), np.zeros((1, mol_node_features))], axis=-1)  # (1, n_residue_types+n_atom_types+1)
    mol_one_hot = np.concatenate([np.zeros((mol_size, protein_node_features+1)), mol_one_hot], axis=-1)  # (n_atoms, n_residue_types+n_atom_types+1)
    one_hot = np.concatenate([protein_one_hot, joint_one_hot, mol_one_hot], axis=0)  # (n_residues+1+n_atoms, n_residue_types+n_atom_types+1)

    # Generate masks
    protein_mask = np.zeros(protein_size + 1 + mol_size)  # (n_residues+1+n_atoms,)
    protein_mask[:protein_size] = 1
    node_mask = np.ones(protein_size + 1 + mol_size)  # (n_residues+1+n_atoms,)

    # Generate edges
    edge_index, edge_attr, edge_mask = get_edges(
        protein_contacts,  # (n_contacts, 3)
        protein_backbone,  # (n_residues-1, 2)
        mol_bonds,        # (n_bonds, 2+bond_features)
        protein_size,     # n_residues
        mol_size         # n_atoms
    )

    # Convert to tensors and add batch dimension
    return {
        'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device).unsqueeze(0),  # (1, n_residues+1+n_atoms, n_residue_types+n_atom_types+1)
        'edge_index': torch.tensor(edge_index, dtype=torch.long, device=device).unsqueeze(0),   # (1, n_edges, 2)
        'edge_attr': torch.tensor(edge_attr, dtype=const.TORCH_FLOAT, device=device).unsqueeze(0),  # (1, n_edges, n_edge_features)
        'node_mask': torch.tensor(node_mask, dtype=const.TORCH_INT, device=device).unsqueeze(0)[:, :, None],  # (1, n_residues+1+n_atoms, 1)
        'edge_mask': torch.tensor(edge_mask, dtype=const.TORCH_INT, device=device).unsqueeze(0)[:, :, None],  # (1, n_edges, 1)
        'protein_mask': torch.tensor(protein_mask, dtype=const.TORCH_INT, device=device).unsqueeze(0)[:, :, None],  # (1, n_residues+1+n_atoms, 1)
    }

def write_pdb_with_predictions(struct, pocket_pred, output_path):
    """Write PDB file with pocket predictions as beta factors.
    
    Args:
        struct: Bio.PDB structure object
        pocket_pred: shape (n_residues,) containing pocket predictions
        output_path: path to output PDB file
    """
    io = PDBIO()
    class PocketSelector(Select):
        def accept_residue(self, residue):
            return True

    io.set_structure(struct)
    io.save(output_path, PocketSelector())

    # Create mapping from residue number to prediction index
    residue_to_idx = {}
    for i, residue in enumerate(struct.get_residues()):
        residue_to_idx[residue.get_id()[1]] = i  # residue.get_id()[1] gives the residue number

    # Update beta factors in output PDB
    with open(output_path, 'r') as f:
        lines = f.readlines()

    # Find all atoms and update their beta factors based on residue number
    for i, line in enumerate(lines):
        if line.startswith('ATOM'):
            try:
                residue_num = int(line[22:26])
                if residue_num in residue_to_idx:
                    pred_idx = residue_to_idx[residue_num]
                    if pred_idx < len(pocket_pred):
                        # Scale the prediction to a reasonable range (e.g., 0-100)
                        beta = float(pocket_pred[pred_idx])
                        lines[i] = line[:60] + f'{beta:6.2f}' + line[66:]
            except ValueError:
                continue  # Skip if residue number is not a valid integer

    with open(output_path, 'w') as f:
        f.writelines(lines)

def predict_pocket(receptor_path, ligand_path, output_path, model, distance_cutoff, device):
    """Predict protein pocket using the model.
    
    Args:
        receptor_path: path to receptor PDB file
        ligand_path: path to ligand SDF file
        output_path: path to output PDB file
        model: YuelPocket model
        distance_cutoff: distance threshold for contacts
        device: torch device
    
    Returns:
        pocket_pred: shape (n_residues,) containing pocket predictions for each residue
    """
    # Load the receptor and ligand
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('', receptor_path)
    ligand = Chem.SDMolSupplier(ligand_path, sanitize=False)[0]

    # Parse protein structure
    protein_pos, protein_one_hot, residue_ids, atom_coords, protein_backbone = parse_protein_structure(struct)

    # Calculate protein contacts
    protein_contacts = calculate_protein_contacts(protein_pos, distance_cutoff)

    # Parse ligand data
    mol_pos, mol_one_hot, mol_bonds = parse_molecule(ligand)

    # Prepare model input
    data = prepare_model_input(
        protein_pos, protein_one_hot, protein_contacts, protein_backbone,
        mol_pos, mol_one_hot, mol_bonds, device
    )

    print("Input shapes:")
    print(data['one_hot'].shape)
    print(data['edge_index'].shape)
    print(data['edge_attr'].shape)
    print(data['node_mask'].shape)
    print(data['edge_mask'].shape)
    print(data['protein_mask'].shape)

    # Get model prediction
    with torch.no_grad():
        pocket_pred = model.forward(data)  # shape: (1, n_residues, 1)
        print("\nPrediction shapes and values:")
        print("pocket_pred shape:", pocket_pred.shape)
        print("pocket_pred min:", pocket_pred.min().item())
        print("pocket_pred max:", pocket_pred.max().item())
        print("pocket_pred mean:", pocket_pred.mean().item())
        
        # Remove batch dimension and squeeze
        pocket_pred = pocket_pred.squeeze()  # shape: (n_residues,)
        print("\nAfter squeezing:")
        print("pocket_pred shape:", pocket_pred.shape)
        print("pocket_pred min:", pocket_pred.min().item())
        print("pocket_pred max:", pocket_pred.max().item())
        print("pocket_pred mean:", pocket_pred.mean().item())

    # Write output PDB with predictions
    write_pdb_with_predictions(struct, pocket_pred, output_path)

    return pocket_pred

def test_from_database():
    """Test pocket prediction using data from database."""
    try:
        # Get a random test case
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pname, lname 
                FROM processed_datasets 
                WHERE split = 'test' 
                ORDER BY RANDOM()
                LIMIT 1
            """)
            pname, lname = cursor.fetchone()
            print(f"Selected test case: {pname} with ligand {lname}")
            
            # Get protein PDB
            cursor.execute("""
                SELECT pdb 
                FROM proteins 
                WHERE name = %s
            """, (pname,))
            pdb_data = cursor.fetchone()[0].tobytes()
            
            # Get ligand SDF
            cursor.execute("""
                SELECT mol 
                FROM ligands 
                WHERE name = %s AND protein_name = %s
            """, (lname, pname))
            sdf_data = cursor.fetchone()[0].tobytes()
            cursor.close()
        
        # Create output directory
        output_dir = 'test_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original files
        protein_path = os.path.join(output_dir, f"{pname}_protein.pdb")
        ligand_path = os.path.join(output_dir, f"{pname}_{lname}_ligand.mol")
        output_path = os.path.join(output_dir, f"{pname}_{lname}_pred.pdb")
        
        # Write original files
        with open(protein_path, 'wb') as f:
            f.write(pdb_data)
        with open(ligand_path, 'wb') as f:
            f.write(sdf_data)
        
        # Set up device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = './models/moad_bs8_date27-05_time09-32-59.692193/last.ckpt'  # Update this path to your model checkpoint
        model = YuelPocket.load_from_checkpoint(model_path, map_location=device).eval().to(device)
        
        # Run prediction
        predict_pocket(
            protein_path,
            ligand_path,
            output_path,
            model,
            distance_cutoff=10.0,
            device=device
        )
        
        print(f"Files saved:")
        print(f"Protein: {protein_path}")
        print(f"Ligand: {ligand_path}")
        print(f"Prediction: {output_path}")
            
    except Exception as e:
        print(f"Error in test_from_database: {str(e)}")
        raise e

def main(args):
    # Set random seed if provided
    if args.random_seed is not None:
        set_deterministic(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yuel_pocket = YuelPocket.load_from_checkpoint(args.model, map_location=device).eval().to(device)
    predict_pocket(args.receptor, args.ligand, args.output, yuel_pocket, args.distance_cutoff, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'receptor', action='store', type=str, 
        help='Path to the receptor PDB file'
    )
    parser.add_argument(
        'ligand', action='store', type=str,
        help='Path to the ligand SDF file'
    )
    parser.add_argument(
        'output', action='store', type=str,
        help='Path to the output PDB file'
    )
    parser.add_argument(
        '--model', action='store', type=str, default='./models/moad_bs8_date27-05_time09-32-59.692193/last.ckpt',
        help='Path to the YuelPocket model checkpoint'
    )
    parser.add_argument(
        '--distance_cutoff', action='store', type=float, required=False, default=10.0,
        help='Distance cutoff for contact prediction (in Angstroms)'
    )
    parser.add_argument(
        '--random_seed', action='store', type=int, required=False, default=None,
        help='Random seed'
    )
    args = parser.parse_args()
    main(args)

    # test_from_database()
