#%%

import argparse
import os
import numpy as np
import torch
from rdkit import Chem
import io
from src import const
from src.datasets import parse_molecule, get_edges
from src.lightning import YuelPocket
from src.utils import set_deterministic
from src.db_utils import db_connection
from src.pdb_utils import Structure
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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

    # Iterate through first model only
    for chain in struct[0]:
        for residue in chain:
            if residue.res_name in ['HOH', 'WAT', 'TIP3', 'H2O', 'SOL']:
                continue
                
            # Get CA atom
            ca_atom = residue.get_atom('CA')
            if ca_atom is None:
                continue
                
            # Get all atom coordinates for this residue
            residue_atoms = residue.get_atoms()
            atom_coords.extend([atom.get_coord() for atom in residue_atoms])
            residue_ids.extend([len(protein_pos)] * len(residue_atoms))
            
            # Get CA position
            pos = ca_atom.get_coord().tolist()
            protein_pos.append(pos)
            protein_one_hot.append(aa_one_hot(residue.res_name))

            # Check if this residue is continuous with the previous one
            if prev_residue is not None:
                # Get residue numbers
                prev_num = prev_residue.res_id
                curr_num = residue.res_id
                
                # Check if residues are in the same chain and consecutive
                if (prev_residue.chain_id == residue.chain_id and  # Same chain
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
            one_hot: shape (n_residues+1+n_atoms, n_residue_types+1+n_atom_types)
            edge_index: shape (n_edges, 2)
            edge_attr: shape (n_edges, n_edge_features)
            node_mask: shape (n_residues+1+n_atoms,)
            edge_mask: shape (n_edges,)
            protein_mask: shape (n_residues+1+n_atoms,)
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

    # Convert to tensors
    return {
        'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
        'edge_index': torch.tensor(edge_index, dtype=torch.long, device=device),
        'edge_attr': torch.tensor(edge_attr, dtype=const.TORCH_FLOAT, device=device),
        'node_mask': torch.tensor(node_mask, dtype=const.TORCH_INT, device=device),
        'edge_mask': torch.tensor(edge_mask, dtype=const.TORCH_INT, device=device),
        'protein_mask': torch.tensor(protein_mask, dtype=const.TORCH_INT, device=device)
    }

def write_pdb_with_predictions(struct, pocket_pred, output_path):
    """Write PDB file with pocket predictions as beta factors.
    
    Args:
        struct: pdb_utils Structure object
        pocket_pred: shape (n_residues,) containing pocket predictions
        output_path: path to output PDB file
    """
    # Create mapping from residue index to prediction index
    residue_to_idx = {}
    for i, residue in enumerate(struct[0].get_residues()):
        residue_to_idx[residue.res_id] = i

    # Update beta factors in all atoms
    for chain in struct[0]:
        for residue in chain:
            if residue.res_id in residue_to_idx:
                pred_idx = residue_to_idx[residue.res_id]
                if pred_idx < len(pocket_pred):
                    # Update beta factor for all atoms in this residue
                    beta = float(pocket_pred[pred_idx])
                    for atom in residue:
                        atom.temp_factor = beta

    # Write the structure to file
    struct.write(output_path)

def _prepare_model_input(pdb_content, sdf_content, distance_cutoff, device):
    """Predict protein pocket using the model with string content.
    
    Args:
        pdb_content: string content of PDB file
        sdf_content: string content of SDF file
        model: YuelPocket model
        distance_cutoff: distance threshold for contacts
        device: torch device
    
    Returns:
        tuple: (pocket_pred, struct) where:
            - pocket_pred: shape (n_residues,) containing pocket predictions
            - struct: Bio.PDB structure object for later use
    """
    # Load the receptor and ligand from string content
    struct = Structure(io.StringIO(pdb_content), skip_hetatm=False, skip_water=True)
    
    # Use RDKit's direct string reading
    ligand = Chem.MolFromMolBlock(sdf_content, sanitize=False, strictParsing=False)
    # Parse protein structure
    protein_pos, protein_one_hot, residue_ids, atom_coords, protein_backbone = parse_protein_structure(struct)
    # print(len(protein_pos), len(protein_one_hot), len(residue_ids), len(atom_coords), len(protein_backbone))
    # Calculate protein contacts
    protein_contacts = calculate_protein_contacts(protein_pos, distance_cutoff)
    # Parse ligand data
    mol_pos, mol_one_hot, mol_bonds = parse_molecule(ligand)
    # Prepare model input
    data = prepare_model_input(
        protein_pos, protein_one_hot, protein_contacts, protein_backbone,
        mol_pos, mol_one_hot, mol_bonds, device
    )
    return data, struct

def _predict_pocket(data, model):

    # Get model prediction
    with torch.no_grad():
        pocket_pred = model.forward(data)  # shape: (1, n_residues, 1)
        # Remove batch dimension and squeeze
        pocket_pred = pocket_pred.squeeze()  # shape: (n_residues,)

    # print(len(pocket_pred))

    return pocket_pred

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
    # Read file contents as text
    with open(receptor_path, 'r') as f:
        pdb_content = f.read()
    with open(ligand_path, 'r') as f:
        sdf_content = f.read()
    
    # Run prediction
    data, struct = _prepare_model_input(pdb_content, sdf_content, distance_cutoff, device)
    pocket_pred = _predict_pocket(data, model)

    # Write output PDB with predictions
    write_pdb_with_predictions(struct, pocket_pred, output_path)

    return pocket_pred

class PdbSdfDataset(torch.utils.data.Dataset):
    """Dataset for pocket prediction from PDB and SDF content.
    
    Args:
        data_pairs: List of (pdb_content, sdf_content) tuples
        distance_cutoff: distance threshold for contacts (default: 10.0)
        device: torch device to put tensors on
    """
    def __init__(self, data_pairs, pdb_cb=None, sdf_cb=None, distance_cutoff=10.0, device='cpu'):
        self.data_pairs = data_pairs
        self.distance_cutoff = distance_cutoff
        self.device = device
        self.pdb_cb = pdb_cb
        self.sdf_cb = sdf_cb
        
    def __len__(self):
        return len(self.data_pairs)
        
    def __getitem__(self, idx):
        # Get data for this index
        pdb_content, sdf_content = self.data_pairs[idx]
        if self.pdb_cb is not None:
            pdb_content = self.pdb_cb(pdb_content)
        if self.sdf_cb is not None:
            sdf_content = self.sdf_cb(sdf_content)
        # Prepare model input using existing functions
        data, _ = _prepare_model_input(
            pdb_content, 
            sdf_content,
            self.distance_cutoff,
            self.device
        )
        return data

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

            pname, lname = '4oc3', '4oc3_2'
            # pname, lname = '6j39', '6j39_0'
            # pname, lname = '2wbk', '2wbk_0'
            
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
        model_path = './models/moad_bs8_date06-06_time11-19-53.016800/last.ckpt'  # Update this path to your model checkpoint
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
    print("Using device: ", device)
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
