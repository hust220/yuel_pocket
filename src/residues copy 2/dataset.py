#%%
import os
import numpy as np
import pickle
import torch
from multiprocessing import Pool
import time
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
from src import const
from src.pdb_utils import Structure
from src import const
from src.pdb_utils import Structure
from src.cache import FileCache
from src.graph import Graph, batch as graph_batch
from io import StringIO
import pyarrow.parquet as pq
from zipfile import ZipFile
import logging
from scipy.spatial import KDTree

# Configure paths
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_ROOT, 'data', 'plinder', 'data', '2024-06', 'v2')
SYSTEMS_DIR = os.path.join(DATA_DIR, 'systems')
SPLIT_PATH = os.path.join(DATA_DIR, 'splits', 'split.parquet')

def generate_sphere_points(n: int) -> np.ndarray:
    """Generate n uniformly distributed points on a unit sphere using Fibonacci lattice."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.stack([x, y, z], axis=1)

def get_sas_points_shrake_rupley(structure: Structure, probe_radius: float = 1.4, n_points_per_atom: int = 96, target_points: int = 50):
    """Generate solvent accessible surface points using Shrake-Rupley algorithm.
    
    Args:
        structure: Structure object
        probe_radius: Probe radius (default 1.4A for water)
        n_points_per_atom: Number of points to sample per atom
        target_points: Stop after finding this many surface points. If None, compute all.
        
    Returns:
        tuple: (points, scores) 
               Note: 'scores' here will be dummy values.
    """
    atoms = structure[0].get_atoms()
    # Handle empty structure case
    if not atoms:
        return np.empty((0, 3)), np.empty((0,))

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
    
    # Determine processing order and limit
    if target_points is None:
        atom_indices = np.arange(n_atoms)
    else:
        atom_indices = np.random.permutation(n_atoms)
    
    processed_count = 0
    
    for i in atom_indices:
        # Stop if we have enough points (only if limitation is set)
        if target_points is not None and processed_count >= target_points:
            break
            
        center = coords[i]
        r = expanded_radii[i]
        
        # Generate points for this atom
        atom_sphere_points = center + unit_sphere * r
        
        # Find neighbors that could overlap
        max_possible_r = np.max(expanded_radii)
        neighbor_indices = tree.query_ball_point(center, r + max_possible_r)
        
        # Filter neighbors: exclude self
        neighbor_indices = [idx for idx in neighbor_indices if idx != i]
        
        if not neighbor_indices:
            sas_points.append(atom_sphere_points)
            processed_count += len(atom_sphere_points)
            continue
            
        neighbor_coords = coords[neighbor_indices]
        neighbor_radii = expanded_radii[neighbor_indices]
        
        # Vectorized check for point occlusion
        P = atom_sphere_points[:, np.newaxis, :]
        N = neighbor_coords[np.newaxis, :, :]
        R_sq = neighbor_radii**2
        
        dists_sq = np.sum((P - N)**2, axis=2)
        is_blocked = np.any(dists_sq < R_sq[np.newaxis, :] - 1e-6, axis=1)
        
        valid_points = atom_sphere_points[~is_blocked]
        if len(valid_points) > 0:
            sas_points.append(valid_points)
            processed_count += len(valid_points)
            
    if not sas_points:
        return np.empty((0, 3)), np.empty((0,))
        
    all_points = np.vstack(sas_points)
    scores = np.ones(len(all_points))
    
    return all_points, scores

class Featurizer:
    @staticmethod
    def atom_one_hot(atom):
        n = const.N_ATOM_TYPES
        one_hot = np.zeros(n)
        if atom not in const.ATOM2IDX:
            atom = 'X'
        one_hot[const.ATOM2IDX[atom]] = 1
        return one_hot

    @staticmethod
    def aa_one_hot(residue):
        n = const.N_RESIDUE_TYPES
        one_hot = np.zeros(n)
        if residue not in const.RESIDUE2IDX:
            residue = 'UNK'
        one_hot[const.RESIDUE2IDX[residue]] = 1
        return one_hot

    @staticmethod
    def bond_one_hot(bond):
        one_hot = [0] * const.N_RDBOND_TYPES
        bond_type = bond.GetBondType()
        if bond_type not in const.RDBOND2IDX:
            bond_type = Chem.rdchem.BondType.ZERO
        one_hot[const.RDBOND2IDX[bond_type]] = 1
        return one_hot

def parse_molecule(mol):
    atom_one_hots = []
    non_h_indices = []
    
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() != 'H':
            atom_one_hots.append(Featurizer.atom_one_hot(atom.GetSymbol()))
            non_h_indices.append(idx)

    if mol.GetNumConformers() == 0:
        positions = np.zeros((len(non_h_indices), 3))
    else:
        positions = mol.GetConformer().GetPositions()[non_h_indices]

    bonds = []
    old_idx_to_new = {old: new for new, old in enumerate(non_h_indices)}
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in old_idx_to_new and j in old_idx_to_new:
            one_hot = Featurizer.bond_one_hot(bond)
            u, v = old_idx_to_new[i], old_idx_to_new[j]
            bonds.append([u, v] + one_hot)
            # bonds.append([v, u] + one_hot)

    return positions, np.array(atom_one_hots), np.array(bonds)

from src.residues.config import get_config

def parse_protein(receptor_pdb):
    structure = Structure()
    structure.read(StringIO(receptor_pdb))
    
    protein_pos = []
    protein_one_hot = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.res_name in const.ALLOWED_RESIDUE_TYPES:
                    ca = residue.get_atom('CA')
                    if ca:
                        protein_pos.append(ca.get_coord())
                        protein_one_hot.append(Featurizer.aa_one_hot(residue.res_name))
        break # Only use first model

    protein_pos = np.array(protein_pos)
    protein_one_hot = np.array(protein_one_hot)
    
    # Calculate SAS points for negative sampling
    # Use config parameters
    config = get_config()
    probe_radius = config.get('sas_probe_radius', 1.4)
    n_points = config.get('sas_n_points', 15)
    
    try:
        sas_points, _ = get_sas_points_shrake_rupley(structure, probe_radius=probe_radius, n_points_per_atom=n_points)
    except Exception as e:
        print(f"Warning: SAS calculation failed: {e}")
        sas_points = np.empty((0, 3))
    
    n_res = len(protein_pos)
    if n_res == 0:
         return np.array([]), np.array([]), np.zeros((0, 3)), np.zeros((0, 2)), np.empty((0, 3))

    # Compute Protein Contacts (CA-CA < 8.0 Angstrom)
    diff = protein_pos[:, None, :] - protein_pos[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    contact_indices = np.where(dists < 8.0)
    protein_contacts = []
    for i, j in zip(*contact_indices):
        if i != j: # Exclude self-loops
             protein_contacts.append([i+1, j+1, dists[i, j]]) 
    
    protein_contacts = np.array(protein_contacts)
    if len(protein_contacts) == 0:
         protein_contacts = np.zeros((0, 3))

    # Compute Protein Backbone
    protein_backbone = []
    for i in range(n_res - 1):
         protein_backbone.append([i, i+1]) 
    
    protein_backbone = np.array(protein_backbone)
    if len(protein_backbone) == 0:
        protein_backbone = np.zeros((0, 2))
        
    return protein_pos, protein_one_hot, protein_contacts, protein_backbone, sas_points

def build_graph_edges(protein_contacts, protein_backbone, mol_bonds, p_size, m_size):
    """Builds edge indices and attributes for the heterogeneous graph."""
    if len(protein_contacts) == 0: protein_contacts = np.empty((0, 3))
    if len(protein_backbone) == 0: protein_backbone = np.empty((0, 2))
    
    # n_bond_feats = mol_bonds.shape[1] - 2 if len(mol_bonds) > 0 else 0
    # Use constant to ensure fixed dimension regardless of bond presence
    n_bond_feats = const.N_RDBOND_TYPES
    # Edge features: [dist, backbone_neighbor, is_prot_joint, is_joint_lig, ...bond_one_hot]
    # Total dim: 4 + n_bond_feats
    
    edges = []  # List of records: (u, v, attr_list)

    # 1. Protein-Protein Contacts: [dist, 0, 0, 0, ...0]
    for u, v, dist in protein_contacts:
        edges.append((int(u)-1, int(v)-1, [dist, 0, 0, 0] + [0] * n_bond_feats))

    # 2. Protein Backbone: [0, 1, 0, 0, ...0]
    # Check for existing edges to update, else add new
    existing_edges = {(e[0], e[1]): idx for idx, e in enumerate(edges)}
    
    for u, v in protein_backbone:
        u, v = int(u), int(v)
        if (u, v) in existing_edges:
            edges[existing_edges[(u, v)]][2][1] = 1
        elif (v, u) in existing_edges:
            edges[existing_edges[(v, u)]][2][1] = 1
        else:
            edges.append((u, v, [0, 1, 0, 0] + [0] * n_bond_feats))

    # 3. Protein-Joint Edges: [0, 0, 1, 0, ...0]
    # Joint node index is exactly `p_size`
    joint_idx = p_size
    for i in range(p_size):
        edges.append((i, joint_idx, [0, 0, 1, 0] + [0] * n_bond_feats))

    # 4. Joint-Ligand Edges: [0, 0, 0, 1, ...0]
    # Ligand nodes start at `p_size + 1`
    for i in range(m_size):
        lig_idx = p_size + 1 + i
        edges.append((joint_idx, lig_idx, [0, 0, 0, 1] + [0] * n_bond_feats))

    # 5. Ligand-Ligand Bonds: [0, 0, 0, 0, ...bond_feats]
    if len(mol_bonds) > 0:
        for bond in mol_bonds:
            u, v = int(bond[0]) + p_size + 1, int(bond[1]) + p_size + 1
            feats = bond[2:].tolist()
            edges.append((u, v, [0, 0, 0, 0] + feats))

    if not edges:
        return [], [], []

    edge_index = [[e[0], e[1]] for e in edges]
    edge_attr = [e[2] for e in edges]
    edge_mask = np.ones(len(edges))

    return edge_index, edge_attr, edge_mask

def build_graph(protein_name, ligand_name, mol_pos, mol_h, mol_bonds, prot_pos, prot_h, prot_contacts, prot_backbone, sas_points):
    """Processes raw parsed data into model-ready tensors."""
    p_name, l_name = protein_name, ligand_name

    m_size, m_feat_dim = mol_h.shape
    p_size, p_feat_dim = prot_h.shape

    if p_size > 1000: return None # Skip large proteins

    # 1. Feature Padding & Concatenation
    # [Protein, Joint, Ligand]
    # Protein: [feat, 0...0]
    # Joint:   [0...0, 1, 0...0]
    # Ligand:  [0...0, feat]
    
    prot_h_padded = np.concatenate([prot_h, np.zeros((p_size, m_feat_dim + 1))], axis=-1)
    joint_h = np.concatenate([np.zeros((1, p_feat_dim)), np.ones((1, 1)), np.zeros((1, m_feat_dim))], axis=-1)
    mol_h_padded = np.concatenate([np.zeros((m_size, p_feat_dim + 1)), mol_h], axis=-1)
    
    h = np.concatenate([prot_h_padded, joint_h, mol_h_padded], axis=0) # [N, Dim]

    # 2. Coordinate Construction
    # Center coordinates on Protein Mean
    p_center = np.mean(prot_pos, axis=0, keepdims=True)
    prot_pos_centered = prot_pos - p_center
    
    # Joint GT must also be shifted relative to new center
    joint_x_raw = np.mean(mol_pos, axis=0, keepdims=True) if len(mol_pos) > 0 else np.zeros((1, 3))
    joint_x_centered = joint_x_raw - p_center
    
    mol_x = np.zeros((m_size, 3)) # Relative/Zero coords for ligand atoms as requested
    
    x = np.concatenate([prot_pos_centered, joint_x_centered, mol_x], axis=0)

    # 3. Mask Construction
    # Masks: [Protein, Joint, Ligand]
    total_nodes = p_size + 1 + m_size
    p_mask, j_mask, l_mask = np.zeros(total_nodes), np.zeros(total_nodes), np.zeros(total_nodes)

    p_mask[:p_size] = 1
    j_mask[p_size] = 1
    l_mask[p_size+1:] = 1

    # 4. Pocket Identification (Residues close to ligand center)
    # Define pocket as residues within 10.0 Angstroms of ligand centroid
    if len(mol_pos) > 0:
        mol_center = np.mean(mol_pos, axis=0)
        dists_to_center = np.linalg.norm(prot_pos - mol_center, axis=1)
        is_pocket_prot = (dists_to_center < 10.0).astype(int)
    else:
        is_pocket_prot = np.zeros(p_size)
    
    # Pad with 0 for joint (1) and ligand (m_size)
    is_pocket = np.concatenate([is_pocket_prot, np.array([0]), np.zeros(m_size)])

    # 5. Negative Pocket Identification (Residues close to random SAS point)
    is_decoy_prot = np.zeros(p_size)
    if sas_points is not None and len(sas_points) > 0:
        # Try finding a point distinct from the true pocket
        for _ in range(20):
            idx = np.random.randint(len(sas_points))
            center = sas_points[idx]
            dists = np.linalg.norm(prot_pos - center, axis=1)
            candidate = (dists < 10.0).astype(int)
            
            # Use dot product as 'overlap' check or simple equality
            # If candidate is not empty and not identical to pocket
            if np.sum(candidate) > 0:
                if np.sum(np.abs(candidate - is_pocket_prot)) > 2: # At least 2 residues different
                     is_decoy_prot = candidate
                     break
    
    is_decoy = np.concatenate([is_decoy_prot, np.array([0]), np.zeros(m_size)])


    # 6. Edge Construction
    edge_index, edge_attr, _ = build_graph_edges(prot_contacts, prot_backbone, mol_bonds, p_size, m_size)

    # 7. Graph Construction
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    num_nodes = len(h)
    
    g = Graph(edge_index_t, num_nodes)
    
    # Add node data
    g.ndata['h'] = torch.tensor(h, dtype=const.TORCH_FLOAT)
    g.ndata['x'] = torch.tensor(x, dtype=const.TORCH_FLOAT)
    g.ndata['protein_mask'] = torch.tensor(p_mask, dtype=const.TORCH_INT)
    g.ndata['joint_mask'] = torch.tensor(j_mask, dtype=const.TORCH_INT)
    g.ndata['ligand_mask'] = torch.tensor(l_mask, dtype=const.TORCH_INT)
    g.ndata['is_pocket'] = torch.tensor(is_pocket, dtype=const.TORCH_FLOAT) # Float for BCE/MSE loss capability
    g.ndata['is_decoy'] = torch.tensor(is_decoy, dtype=const.TORCH_FLOAT)
    
    # Add edge data
    if len(edge_attr) > 0:
        g.edata['e'] = torch.tensor(edge_attr, dtype=const.TORCH_FLOAT)
    
    g.pname = p_name
    g.lname = l_name
    return g

def collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return graph_batch(batch)

class PocketDataset(Dataset):
    collate_fn = staticmethod(collate)
    def __init__(self, device=None, progress_bar=True, split='train', limit=None):
        self.progress_bar = progress_bar
        self.device = device
        self.split = split
        
        # Determine split filter
        split_filter = split
        if split == 'validation':
            split_filter = 'val'
        
        start_time = time.time()
        print(f'Loading dataset IDs from split file (split={split})...')
        table = pq.read_table(SPLIT_PATH, columns=['system_id', 'split'])
        df = table.to_pandas()
        filtered_df = df[df['split'] == split_filter]
        if limit:
            filtered_df = filtered_df.head(limit)
        self.ids = filtered_df['system_id'].tolist()
        print('Loaded dataset IDs, Time: ', time.time() - start_time, 's')
        self.valid_indices = np.arange(len(self.ids))
        print(f"Found {len(self.valid_indices)} valid entries")

    def _read_from_zip(self, system_id):
        zip_path = os.path.join(SYSTEMS_DIR, f"{system_id[1:3]}.zip")
        if not os.path.exists(zip_path):
             # Try failover or search? For now fail hard or return None
             raise FileNotFoundError(f"Zip bucket {zip_path} not found for {system_id}")

        with ZipFile(zip_path, 'r') as zf:
            receptor_path = f"{system_id}/receptor.pdb"
            with zf.open(receptor_path) as f:
                receptor_pdb = f.read().decode("utf-8", "ignore")

            parts = system_id.split("__")
            candidate_ligands = []
            
            ligands_str = parts[3]
            ligand_ids = ligands_str.split("_")
            
            for lig_id in ligand_ids:
                sdf_path = f"{system_id}/ligand_files/{lig_id}.sdf"
                with zf.open(sdf_path) as f:
                    sdf_content = f.read().decode("utf-8", "ignore")
                    candidate_ligands.append(sdf_content)
            
            # Select the largest ligand by atom count
            best_mol = None
            max_atoms = -1
            
            for sdf in candidate_ligands:
                mol = Chem.MolFromMolBlock(sdf, sanitize=False)
                if mol is not None:
                    n_atoms = mol.GetNumAtoms()
                    if n_atoms > max_atoms:
                        max_atoms = n_atoms
                        best_mol = mol
            
            return system_id, receptor_pdb, best_mol

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, item):
        # Map the requested index to the actual index in the dataset
        actual_idx = self.valid_indices[item]
        item_id = self.ids[actual_idx]
        
        # Get the sample data from ZIP directly
        system_id = item_id 
        
        raw_system_id, receptor_pdb, ligand_mol = self._read_from_zip(system_id)
        
        if ligand_mol is None: # Should be caught by read_from_zip but safe to check
             raise ValueError(f"Ligand mol is None for {system_id}")

        mol_pos, mol_one_hot, mol_bonds = parse_molecule(ligand_mol)
        protein_pos, protein_one_hot, protein_contacts, protein_backbone, sas_points = parse_protein(receptor_pdb)

        if len(protein_pos) == 0:
              raise ValueError(f"No valid residues found for {system_id}")

        result = build_graph(
            protein_name=raw_system_id,
            ligand_name=raw_system_id,
            mol_pos=mol_pos,
            mol_h=mol_one_hot,
            mol_bonds=mol_bonds,
            prot_pos=protein_pos,
            prot_h=protein_one_hot,
            prot_contacts=protein_contacts,
            prot_backbone=protein_backbone,
            sas_points=sas_points
        )
        
        if result is None:
            return None
            
        return result



def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

