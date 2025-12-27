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
from zipfile import ZipFile, BadZipFile
import logging
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from io import BytesIO
from .config import get_config

# Configure paths
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_ROOT, 'data', 'plinder', 'data', '2024-06', 'v2')
SYSTEMS_DIR = os.path.join(DATA_DIR, 'systems')
SPLIT_PATH = os.path.join(DATA_DIR, 'splits', 'split.parquet')
SAS_PATH = os.path.join(PROJ_ROOT, 'data', 'plinder', 'sas_points.zip')

# Extend allowed residue types locally
POS_SC_ALLOWED_RESIDUE_TYPES = const.ALLOWED_RESIDUE_TYPES + ['BB']
POS_SC_RESIDUE2IDX = {res: idx for idx, res in enumerate(POS_SC_ALLOWED_RESIDUE_TYPES)}
POS_SC_N_RESIDUE_TYPES = len(POS_SC_ALLOWED_RESIDUE_TYPES)

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
        n = POS_SC_N_RESIDUE_TYPES
        one_hot = np.zeros(n)
        if residue not in POS_SC_RESIDUE2IDX:
            residue = 'UNK'
        one_hot[POS_SC_RESIDUE2IDX[residue]] = 1
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


def parse_protein(receptor_pdb):
    structure = Structure()
    structure.read(StringIO(receptor_pdb))
    
    protein_pos = []
    protein_one_hot = []
    
    # Standard backbone atom names + Hydrogens to exclude from sidechain
    backbone_atoms = {'N', 'CA', 'C', 'O', 'H', 'HA', 'HA2', 'HA3', 'OXT'}
    
    for model in structure:
        for chain in model:
            sorted_residues = sorted(chain.residues, key=lambda r: r.res_id)
            for residue in sorted_residues:
                if residue.res_name in const.ALLOWED_RESIDUE_TYPES:
                    ca = residue.get_atom('CA')
                    if ca:
                        # --- Node 1: CA (Unified Encoding) ---
                        ca_coord = ca.get_coord()
                        protein_pos.append(ca_coord)
                        # Use a unified encoding for CA (Specific 'BB' token)
                        protein_one_hot.append(Featurizer.aa_one_hot('BB')) 
                        
                        # --- Node 2: Sidechain Center (AA Encoding) ---
                        sc_atoms = [a for a in residue.atoms if a.atom_name not in backbone_atoms]
                        
                        if len(sc_atoms) > 0:
                            sc_coords = np.array([a.get_coord() for a in sc_atoms])
                            sc_center = np.mean(sc_coords, axis=0)
                        else:
                            # Fallback for GLY or missing sidechain: use CA position
                            sc_center = ca_coord
                            
                        protein_pos.append(sc_center)
                        protein_one_hot.append(Featurizer.aa_one_hot(residue.res_name))
                        
        break # Only use first model

    protein_pos = np.array(protein_pos)
    protein_one_hot = np.array(protein_one_hot)
    
    # Calculate SAS points for negative sampling
    # config = get_config()
    # probe_radius = config.get('sas_probe_radius', 1.4)
    # n_points = config.get('sas_n_points', 15)
    
    # try:
    #     sas_points, _ = get_sas_points_shrake_rupley(structure, probe_radius=probe_radius, n_points_per_atom=n_points)
    # except Exception as e:
    #     print(f"Warning: SAS calculation failed: {e}")
    sas_points = np.empty((0, 3))
    
    n_nodes = len(protein_pos)
    if n_nodes == 0:
         return np.array([]), np.array([]), np.zeros((0, 3)), np.zeros((0, 2)), np.empty((0, 3))

    # Compute Protein Contacts (Node-Node < 8.0 Angstrom)
    # This now runs on all CA and SC nodes
    if n_nodes > 1:
        # KDTree for efficiency with 2x nodes
        tree = KDTree(protein_pos)
        # Find all pairs within 8.0
        # query_pairs returns set of (i, j) where i < j
        contact_pairs = tree.query_pairs(r=8.0)
        
        protein_contacts = []
        for i, j in contact_pairs:
             dist = np.linalg.norm(protein_pos[i] - protein_pos[j])
             protein_contacts.append([i, j, dist])
        
        protein_contacts = np.array(protein_contacts)
    else:
        protein_contacts = np.zeros((0, 3))
        
    if len(protein_contacts) == 0:
         protein_contacts = np.zeros((0, 3))

    if len(protein_contacts) == 0:
         protein_contacts = np.zeros((0, 3))

    return protein_pos, protein_one_hot, protein_contacts, sas_points

def build_graph_edges(protein_contacts, p_size, prot_pos, probes):
    """Builds edge indices and attributes for the heterogeneous graph."""
    if len(protein_contacts) == 0: protein_contacts = np.empty((0, 3))
    
    # Edge features: [dist]
    # Total dim: 1
    
    edges = []  # List of records: (u, v, attr_list)

    # 1. Protein-Protein Contacts: [dist]
    for u, v, dist in protein_contacts:
        edges.append((int(u), int(v), [dist]))

    # 6. Probe-Protein Edges: [dist]
    num_complex_edges = len(edges)
    
    if len(probes) > 0:
        probe_start = p_size
        
        # Use KDTree for efficient radius search
        tree = KDTree(prot_pos)
        queries = tree.query_ball_point(probes, r=10.0)
        
        for i, neighbors in enumerate(queries):
            probe_node_idx = probe_start + i
            probe_coord = probes[i]
            
            # Connect to neighbors
            if len(neighbors) > 0:
                p_coords = prot_pos[neighbors]
                dists = np.linalg.norm(p_coords - probe_coord, axis=1)
                
                for p_idx, dist in zip(neighbors, dists):
                    edges.append((int(p_idx), probe_node_idx, [dist]))

    if not edges:
        return [], [], [], []

    edge_index = [[e[0], e[1]] for e in edges]
    edge_attr = [e[2] for e in edges]
    
    complex_mask = np.zeros(len(edges))
    complex_mask[:num_complex_edges] = 1.0
    
    probe_mask = np.zeros(len(edges))
    probe_mask[num_complex_edges:] = 1.0

    return edge_index, edge_attr, complex_mask, probe_mask

def build_graph(protein_name, ligand_name, mol_pos, mol_h, mol_bonds, prot_pos, prot_h, prot_contacts, sas_points, pick_samples=True):
    """Processes raw parsed data into model-ready tensors.
    
    Args:
        pick_samples (bool): If True, selects 1 pocket and 1 decoy probe (Training mode). 
                             If False, uses all sas_points as probes (Inference mode).
    """
    p_name, l_name = protein_name, ligand_name

    p_size, p_feat_dim = prot_h.shape

    # p_size is now 2 * n_residues
    # Threshold check roughly adapted (500 residues -> 1000 nodes)
    if pick_samples and p_size > 2000: return None # Skip large proteins during training

    # Coordinate Construction
    # Center coordinates on Protein Mean
    p_center = np.mean(prot_pos, axis=0, keepdims=True)
    prot_pos_centered = prot_pos - p_center
    
    # Select SAS Points / Probes
    selected_probes = np.zeros((0, 3))
    is_training_sample = False
    
    if pick_samples and len(mol_pos) > 0 and sas_points is not None and len(sas_points) > 0:
        # Training Mode: Pick 2 samples
        dists = cdist(sas_points, mol_pos).min(axis=1)
        pocket_indices = np.where(dists < 3.0)[0]
        decoy_indices = np.where(dists >= 3.0)[0]
        
        p_sas = np.zeros(3)
        d_sas = np.zeros(3)
        
        if len(pocket_indices) > 0:
            p_idx = np.random.choice(pocket_indices)
            p_sas = sas_points[p_idx]
        else:
             # Fallback to closest
            p_idx = np.argmin(dists)
            p_sas = sas_points[p_idx]
            
        if len(decoy_indices) > 0:
            d_idx = np.random.choice(decoy_indices)
            d_sas = sas_points[d_idx]
        else:
            d_idx = np.argmax(dists)
            d_sas = sas_points[d_idx]
            
        selected_probes = np.stack([p_sas, d_sas])
        is_training_sample = True
        
    elif not pick_samples and sas_points is not None:
        # Inference Mode: Use all points
        selected_probes = sas_points
    
    # Center probes
    if len(selected_probes) > 0:
        probes_centered = selected_probes - p_center
    else:
        probes_centered = np.zeros((0, 3))
    
    # Probe Features: [0...0]
    n_probes = len(selected_probes)
    probe_h = np.zeros((n_probes, p_feat_dim))
    
    h = np.concatenate([prot_h, probe_h], axis=0) # [N, Dim]
    x = np.concatenate([prot_pos_centered, probes_centered], axis=0)

    # 3. Mask Construction
    # Masks: [Protein, Probes...]
    total_nodes = len(h)
    p_mask = np.zeros(total_nodes)
    sas_node_mask = np.zeros(total_nodes)

    p_mask[:p_size] = 1
    
    probe_start = p_size
    sas_node_mask[probe_start:] = 1

    # 4. Pocket/Decoy Identification (Labeling)
    is_pocket = np.zeros(total_nodes)
    is_decoy = np.zeros(total_nodes)
    
    if is_training_sample and n_probes == 2:
        is_pocket[probe_start] = 1.0 # First probe is pocket
        is_decoy[probe_start + 1] = 1.0 # Second probe is decoy

    # 6. Edge Construction
    edge_index, edge_attr, complex_mask, probe_mask = build_graph_edges(
        prot_contacts, p_size, 
        prot_pos_centered, probes_centered
    )

    # 7. Graph Construction
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    num_nodes = len(h)
    
    g = Graph(edge_index_t, num_nodes)
    
    # Add node data
    g.ndata['h'] = torch.tensor(h, dtype=const.TORCH_FLOAT)
    g.ndata['x'] = torch.tensor(x, dtype=const.TORCH_FLOAT)
    g.ndata['protein_mask'] = torch.tensor(p_mask, dtype=const.TORCH_INT)
    g.ndata['sas_mask'] = torch.tensor(sas_node_mask, dtype=const.TORCH_INT)
    g.ndata['is_pocket'] = torch.tensor(is_pocket, dtype=const.TORCH_FLOAT)
    g.ndata['is_decoy'] = torch.tensor(is_decoy, dtype=const.TORCH_FLOAT)

    
    # Add edge data
    if len(edge_attr) > 0:
        g.edata['e'] = torch.tensor(edge_attr, dtype=const.TORCH_FLOAT)
        g.edata['complex_mask'] = torch.tensor(complex_mask, dtype=const.TORCH_FLOAT)
        g.edata['probe_mask'] = torch.tensor(probe_mask, dtype=const.TORCH_FLOAT)
    
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
        self.sas_zip = None # Cache for SAS zip file
        
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
        protein_pos, protein_one_hot, protein_contacts, sas_points = parse_protein(receptor_pdb)

        if len(protein_pos) == 0:
              raise ValueError(f"No valid residues found for {system_id}")
        
        # Load SAS Points from Zip (Lazy Load)
        if self.sas_zip is None:
            if os.path.exists(SAS_PATH):
                self.sas_zip = ZipFile(SAS_PATH, 'r')
            else:
                self.sas_zip = None

        sas_points = None
        if self.sas_zip is not None:
            try:
                with self.sas_zip.open(f"{system_id}.npy") as f:
                    sas_points = np.load(f)
            except KeyError:
                pass # Not found
            except Exception as e:
                pass # print(f"Error reading SAS for {system_id}: {e}")
        
        if sas_points is None:
             sas_points = np.empty((0, 3))

        result = build_graph(
            protein_name=raw_system_id,
            ligand_name=raw_system_id,
            mol_pos=mol_pos,
            mol_h=mol_one_hot,
            mol_bonds=mol_bonds,
            prot_pos=protein_pos,
            prot_h=protein_one_hot,
            prot_contacts=protein_contacts,
            sas_points=sas_points
        )
        
        if result is None:
            return None
            
        return result



def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

