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
LIGANDS_PATH = os.path.join(DATA_DIR, 'fingerprints', 'ligands_per_inchikey.parquet')

def parse_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, None
    
    atom_one_hots = []
    for atom in mol.GetAtoms():
        atom_one_hots.append(Featurizer.atom_one_hot(atom.GetSymbol()))
    
    bonds = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append([u, v, 1])
    
    return np.array(atom_one_hots), np.array(bonds)

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
    
    return protein_pos, protein_one_hot

def build_single_probe_data(probe_coord, prot_pos, prot_h, mol_h, mol_bonds, label_type, cutoff=10.0):
    """Builds raw data for a single probe's subgraph."""
    # 1. Find neighbors (Protein only, Ligand is always fully included as 2D)
    prot_dists = np.linalg.norm(prot_pos - probe_coord, axis=1)
    prot_indices = np.where(prot_dists < cutoff)[0]
    
    m_size = len(mol_h)
    p_size = len(prot_indices)
    num_nodes = p_size + m_size + 1
    probe_local_idx = p_size + m_size
    
    if p_size == 0 and m_size == 0:
        return None

    # 2. Features
    sub_prot_h = prot_h[prot_indices]
    p_feat_dim = prot_h.shape[1]
    m_feat_dim = mol_h.shape[1]
    
    h_prot = np.concatenate([sub_prot_h, np.zeros((p_size, m_feat_dim))], axis=-1)
    h_mol = np.concatenate([np.zeros((m_size, p_feat_dim)), mol_h], axis=-1)
    h_probe = np.zeros((1, p_feat_dim + m_feat_dim))
    h = np.concatenate([h_prot, h_mol, h_probe], axis=0) # [N, TotDim]
    
    # 3. Edges
    edges = [] # (u, v, [dist, is_bond], is_complex, is_probe)
    
    # a. Protein-Protein: Full Connection
    sub_prot_pos = prot_pos[prot_indices]
    for i in range(p_size):
        for j in range(p_size):
            if i < j:
                dist = np.linalg.norm(sub_prot_pos[i] - sub_prot_pos[j])
                edges.append((i, j, [dist, 0], 1.0, 0.0))
                
    # b. Ligand Internal: Original Bonds (Full 2D structure)
    if len(mol_bonds) > 0:
        for bond in mol_bonds:
            u_old, v_old = int(bond[0]), int(bond[1])
            if u_old < m_size and v_old < m_size:
                u_new, v_new = u_old + p_size, v_old + p_size
                edges.append((u_new, v_new, [0.0, 1.0], 1.0, 0.0))

    # c. Protein-Probe: All connected within cutoff
    for i in range(p_size):
        dist = np.linalg.norm(sub_prot_pos[i] - probe_coord)
        edges.append((i, probe_local_idx, [dist, 0.0], 0.0, 1.0))
        
    # d. Ligand-Probe: All connected (2D, no physical distance)
    for i in range(m_size):
        edges.append((i + p_size, probe_local_idx, [0.0, 0.0], 0.0, 1.0))
        
    # 4. Masks & Labels
    ndata = {}
    ndata['protein_mask'] = np.zeros(num_nodes)
    ndata['protein_mask'][:p_size] = 1
    
    ndata['ligand_mask'] = np.zeros(num_nodes)
    ndata['ligand_mask'][p_size:p_size+m_size] = 1
    
    ndata['sas_mask'] = np.zeros(num_nodes)
    ndata['sas_mask'][probe_local_idx] = 1
    
    ndata['is_pocket'] = np.zeros(num_nodes)
    ndata['is_decoy'] = np.zeros(num_nodes)
    ndata['is_decoy2'] = np.zeros(num_nodes)
    
    if label_type == 'is_pocket': ndata['is_pocket'][probe_local_idx] = 1.0
    elif label_type == 'is_decoy': ndata['is_decoy'][probe_local_idx] = 1.0
    elif label_type == 'is_decoy2': ndata['is_decoy2'][probe_local_idx] = 1.0
    
    return h, edges, ndata

def build_graph(protein_name, ligand_name, mol_pos, mol_h, mol_bonds, prot_pos, prot_h, sas_points, mol_decoys=None, pick_samples=True):
    """Processes raw parsed data into model-ready tensors."""
    p_name, l_name = protein_name, ligand_name

    # Select SAS Points / Probes
    selected_probes_info = [] # List of (coord, ligand_data, label_type)
    
    if pick_samples and len(mol_pos) > 0 and sas_points is not None and len(sas_points) > 0:
        # Training Mode: Pick pocket and decoy
        ligand_centroid = np.mean(mol_pos, axis=0, keepdims=True)
        dists = cdist(sas_points, ligand_centroid).flatten()
        pocket_indices = np.where(dists < 4.0)[0]
        decoy_indices = np.where(dists >= 4.0)[0]
        
        p_idx = np.random.choice(pocket_indices) if len(pocket_indices) > 0 else np.argmin(dists)
        p_sas = sas_points[p_idx]
        
        # 1. is_pocket Graph
        selected_probes_info.append((p_sas, mol_h, mol_bonds, 'is_pocket'))
        # 2. is_decoy Graph (Sample 50)
        if len(decoy_indices) > 0:
            n_decoy = min(50, len(decoy_indices))
            sampled_d_indices = np.random.choice(decoy_indices, n_decoy, replace=False)
            for d_idx in sampled_d_indices:
                selected_probes_info.append((sas_points[d_idx], mol_h, mol_bonds, 'is_decoy'))
        else:
            d_idx = np.argmax(dists)
            selected_probes_info.append((sas_points[d_idx], mol_h, mol_bonds, 'is_decoy'))
        # 3. is_decoy2 Graph (Wrong Ligands - Sample 50)
        if mol_decoys:
            for m_h2, m_b2 in mol_decoys:
                selected_probes_info.append((p_sas, m_h2, m_b2, 'is_decoy2'))
            
    elif not pick_samples and sas_points is not None:
        # Inference Mode: All points with Ligand 1
        for p in sas_points:
            selected_probes_info.append((p, mol_h, mol_bonds, 'inference'))
    
    if not selected_probes_info:
        return None

    # Build components and merge manually
    all_h, all_edges, all_ndata = [], [], []
    total_nodes = 0
    
    for probe_coord, m_h, m_bonds, label in selected_probes_info:
        res = build_single_probe_data(probe_coord, prot_pos, prot_h, m_h, m_bonds, label)
        if res is None: continue
        
        h_i, edges_i, ndata_i = res
        
        # Shift edge indices
        for e in edges_i:
            all_edges.append((e[0] + total_nodes, e[1] + total_nodes, e[2], e[3], e[4]))
            
        all_h.append(h_i)
        all_ndata.append(ndata_i)
        total_nodes += len(h_i)

    if not all_edges:
        return None

    # Construct final merged Graph object
    edge_index_t = torch.tensor([[e[0], e[1]] for e in all_edges], dtype=torch.long).t().contiguous()
    edge_attr_t = torch.tensor([e[2] for e in all_edges], dtype=torch.float)
    complex_mask_t = torch.tensor([e[3] for e in all_edges], dtype=torch.float)
    probe_mask_t = torch.tensor([e[4] for e in all_edges], dtype=torch.float)
    
    h_t = torch.tensor(np.concatenate(all_h, axis=0), dtype=torch.float)
    
    g = Graph(edge_index_t, total_nodes)
    g.ndata['h'] = h_t
    
    # Merge ndata
    for key in ['protein_mask', 'ligand_mask', 'sas_mask', 'is_pocket', 'is_decoy', 'is_decoy2']:
        data_list = [d[key] for d in all_ndata]
        g.ndata[key] = torch.tensor(np.concatenate(data_list, axis=0), dtype=torch.float if 'mask' not in key else torch.int)
    
    g.ndata['joint_mask'] = torch.zeros(total_nodes, dtype=torch.int)
    g.edata['e'] = edge_attr_t
    g.edata['complex_mask'] = complex_mask_t
    g.edata['probe_mask'] = probe_mask_t
    
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
        
        # Load Unique Ligand SMILES from PLINDER
        print(f"Loading unique ligands from {LIGANDS_PATH}...")
        self.random_ligand_smiles = []
        if os.path.exists(LIGANDS_PATH):
            table = pq.read_table(LIGANDS_PATH, columns=['ligand_rdkit_canonical_smiles'])
            self.random_ligand_smiles = table['ligand_rdkit_canonical_smiles'].to_pylist()
        print(f"Loaded {len(self.random_ligand_smiles)} SMILES.")

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
        protein_pos, protein_one_hot = parse_protein(receptor_pdb)

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

        # Get random ligands for is_decoy2 (Sample 50)
        mol_decoys = []
        if self.random_ligand_smiles:
            import random
            n_random = min(50, len(self.random_ligand_smiles))
            random_smis = random.sample(self.random_ligand_smiles, n_random)
            for smi in random_smis:
                m_h2, m_b2 = parse_smiles(smi)
                if m_h2 is not None and len(m_h2) > 0:
                    mol_decoys.append((m_h2, m_b2))

        result = build_graph(
            protein_name=raw_system_id,
            ligand_name=raw_system_id,
            mol_pos=mol_pos,
            mol_h=mol_one_hot,
            mol_bonds=mol_bonds,
            prot_pos=protein_pos,
            prot_h=protein_one_hot,
            sas_points=sas_points,
            mol_decoys=mol_decoys
        )
        
        if result is None:
            return None
            
        return result



def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

