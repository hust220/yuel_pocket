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
import random
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

# Configure paths
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_ROOT, 'data', 'plinder', 'data', '2024-06', 'v2')
SYSTEMS_DIR = os.path.join(DATA_DIR, 'systems')
SPLIT_PATH = os.path.join(DATA_DIR, 'splits', 'split.parquet')
LIGANDS_PATH = os.path.join(DATA_DIR, 'fingerprints', 'ligands_per_inchikey.parquet')

# Local residue mapping to include 'BB'
RES2_ALLOWED_RESIDUE_TYPES = const.ALLOWED_RESIDUE_TYPES + ['BB']
RES2_RESIDUE2IDX = {res: idx for idx, res in enumerate(RES2_ALLOWED_RESIDUE_TYPES)}
RES2_N_RESIDUE_TYPES = len(RES2_ALLOWED_RESIDUE_TYPES)

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
        n = RES2_N_RESIDUE_TYPES
        one_hot = np.zeros(n)
        if residue not in RES2_RESIDUE2IDX:
            residue = 'UNK'
        one_hot[RES2_RESIDUE2IDX[residue]] = 1
        return one_hot

    @staticmethod
    def bond_one_hot(bond):
        one_hot = [0] * const.N_RDBOND_TYPES
        if isinstance(bond, int):
             one_hot[0] = 1 
             return one_hot
        bond_type = bond.GetBondType()
        if bond_type not in const.RDBOND2IDX:
            bond_type = Chem.rdchem.BondType.ZERO
        one_hot[const.RDBOND2IDX[bond_type]] = 1
        return one_hot

def parse_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, None, None
    atom_one_hots = []
    for atom in mol.GetAtoms():
        atom_one_hots.append(Featurizer.atom_one_hot(atom.GetSymbol()))
    bonds = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        one_hot = Featurizer.bond_one_hot(bond)
        bonds.append([u, v] + one_hot)
    return np.zeros((len(atom_one_hots), 3)), np.array(atom_one_hots), np.array(bonds)

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
    return positions, np.array(atom_one_hots), np.array(bonds)

def parse_protein(receptor_pdb):
    structure = Structure()
    structure.read(StringIO(receptor_pdb))
    protein_pos = []
    protein_one_hot = []
    
    backbone_atoms = {'N', 'CA', 'C', 'O', 'H', 'HA', 'HA2', 'HA3', 'OXT'}

    for model in structure:
        for chain in model:
            sorted_residues = sorted(chain.residues, key=lambda r: r.res_id)
            for residue in sorted_residues:
                if residue.res_name in const.ALLOWED_RESIDUE_TYPES:
                    ca = residue.get_atom('CA')
                    if ca:
                        # 1. CA Node
                        ca_coord = ca.get_coord()
                        protein_pos.append(ca_coord)
                        protein_one_hot.append(Featurizer.aa_one_hot('BB'))
                        
                        # 2. SC Center Node
                        sc_atoms = [a for a in residue.atoms if a.atom_name not in backbone_atoms]
                        if len(sc_atoms) > 0:
                            sc_coords = np.array([a.get_coord() for a in sc_atoms])
                            sc_center = np.mean(sc_coords, axis=0)
                        else:
                            sc_center = ca_coord
                        protein_pos.append(sc_center)
                        protein_one_hot.append(Featurizer.aa_one_hot(residue.res_name))
        break 
        
    protein_pos = np.array(protein_pos)
    protein_one_hot = np.array(protein_one_hot)
    
    # Compute Contacts (8.0A between all protein nodes) using KDTree for efficiency
    if len(protein_pos) > 0:
        from scipy.spatial import KDTree
        tree = KDTree(protein_pos)
        pairs = tree.query_pairs(r=8.0)
        protein_contacts = []
        for i, j in pairs:
            dist = np.linalg.norm(protein_pos[i] - protein_pos[j])
            # Store only one direction; GNN handles bi-directionality
            protein_contacts.append([i+1, j+1, dist]) 
        protein_contacts = np.array(protein_contacts) if protein_contacts else np.zeros((0, 3))
    else:
        protein_contacts = np.zeros((0, 3))

    return protein_pos, protein_one_hot, protein_contacts

def build_protein_graph(prot_h, prot_pos, prot_contacts):
    p_size = len(prot_h)
    h_extra = np.zeros((1, prot_h.shape[1]))
    h_full = np.concatenate([prot_h, h_extra], axis=0) 
    h_full_padded = np.concatenate([h_full, np.zeros((p_size + 1, const.N_ATOM_TYPES + 1))], axis=-1)
    
    edges = []
    # 1. Internal Contacts
    for u, v, dist in prot_contacts:
        edges.append((int(u)-1, int(v)-1, [dist, 1.0, 0.0, 0.0, 0.0]))
        
    # 2. Connect all normal nodes to extra node (index p_size)
    extra_idx = p_size
    for i in range(p_size):
        edges.append((i, extra_idx, [0.0, 0.0, 0.0, 1.0, 0.0])) 
        
    edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor([e[2] for e in edges], dtype=torch.float) if edges else torch.empty((0, 5))
    
    g = Graph(edge_index, p_size + 1)
    g.ndata['h'] = torch.tensor(h_full_padded, dtype=const.TORCH_FLOAT)
    
    p_mask = torch.ones(p_size + 1, dtype=torch.int) # Now includes extra node
    g.ndata['protein_mask'] = p_mask
    
    e_mask = torch.zeros(p_size + 1, dtype=torch.int)
    e_mask[extra_idx] = 1
    g.ndata['extra_mask'] = e_mask
    
    g.ndata['ligand_mask'] = torch.zeros(p_size + 1, dtype=torch.int)
    # is_pocket is now already the right size (2N) passed from build_graph
    g.ndata['is_pocket'] = torch.zeros(p_size + 1, dtype=torch.float) # Will be set in build_graph or here
    g.ndata['is_true_ligand'] = torch.zeros(p_size + 1, dtype=torch.int)
    g.edata['e'] = edge_attr
    return g

def build_ligand_graph(mol_h, mol_bonds, is_true):
    m_size = len(mol_h)
    h_extra = np.zeros((1, mol_h.shape[1]))
    mol_h_full = np.concatenate([mol_h, h_extra], axis=0)
    h_full_padded = np.concatenate([np.zeros((m_size + 1, RES2_N_RESIDUE_TYPES + 1)), mol_h_full], axis=-1)
    
    edges = []
    if len(mol_bonds) > 0:
        for bond in mol_bonds:
            u, v = int(bond[0]), int(bond[1])
            edges.append((u, v, [0.0, 0.0, 1.0, 0.0, 0.0])) # dist, contact, bond, p_extra, l_extra
            
    extra_idx = m_size
    for i in range(m_size):
        edges.append((i, extra_idx, [0.0, 0.0, 0.0, 0.0, 1.0]))
            
    edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor([e[2] for e in edges], dtype=torch.float) if edges else torch.empty((0, 5))
    
    g = Graph(edge_index, m_size + 1)
    g.ndata['h'] = torch.tensor(h_full_padded, dtype=const.TORCH_FLOAT)
    g.ndata['protein_mask'] = torch.zeros(m_size + 1, dtype=torch.int)
    
    l_mask = torch.ones(m_size + 1, dtype=torch.int) # Now includes extra node
    g.ndata['ligand_mask'] = l_mask

    e_mask = torch.zeros(m_size + 1, dtype=torch.int)
    e_mask[extra_idx] = 1
    g.ndata['extra_mask'] = e_mask
    
    g.ndata['is_pocket'] = torch.zeros(m_size + 1, dtype=torch.float)
    g.ndata['is_true_ligand'] = torch.ones(m_size + 1, dtype=torch.int) if is_true else torch.zeros(m_size + 1, dtype=torch.int)
    g.edata['e'] = edge_attr
    return g

def build_graph(protein_name, ligand_name, mol_pos, mol_h, mol_bonds, prot_pos, prot_h, prot_contacts, extra_ligands=[]):
    p_size = len(prot_h)
    if p_size > 10000: return None # Increased limit for large proteins
    
    if len(mol_pos) > 0:
        from scipy.spatial import KDTree
        lig_tree = KDTree(mol_pos)
        min_dists, _ = lig_tree.query(prot_pos)
        is_pocket = (min_dists < 4.0).astype(float)
    else:
        is_pocket = np.zeros(p_size)
        
    prot_g = build_protein_graph(prot_h, prot_pos, prot_contacts)
    prot_g.ndata['is_pocket'][:p_size] = torch.tensor(is_pocket, dtype=torch.float)
    
    lig_graphs = []
    lig_graphs.append(build_ligand_graph(mol_h, mol_bonds, is_true=True))
    for m_h_ex, m_b_ex in extra_ligands:
        lig_graphs.append(build_ligand_graph(m_h_ex, m_b_ex, is_true=False))
        
    final_g = graph_batch([prot_g] + lig_graphs)
    final_g.pname = protein_name
    final_g.lname = ligand_name
    return final_g

def collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return graph_batch(batch)

class PocketDataset(Dataset):
    collate_fn = staticmethod(collate)
    def __init__(self, device=None, progress_bar=True, split='train', limit=None):
        self.progress_bar = progress_bar
        self.device = device
        self.split = split
        self.cache = FileCache(cache_mode=None, dataset_name=f'pocket_dataset_v4_{split}')
        print(f"Loading unique ligands from {LIGANDS_PATH}...")
        self.random_ligand_smiles = []
        if os.path.exists(LIGANDS_PATH):
             table = pq.read_table(LIGANDS_PATH, columns=['ligand_rdkit_canonical_smiles'])
             self.random_ligand_smiles = table['ligand_rdkit_canonical_smiles'].to_pylist()
        print(f"Loaded {len(self.random_ligand_smiles)} SMILES.")
        split_filter = split
        if split == 'validation': split_filter = 'val'
        print(f'Loading dataset IDs from split file (split={split})...')
        table = pq.read_table(SPLIT_PATH, columns=['system_id', 'split'])
        df = table.to_pandas()
        filtered_df = df[df['split'] == split_filter]
        if limit: filtered_df = filtered_df.head(limit)
        self.ids = filtered_df['system_id'].tolist()
        self.valid_indices = np.arange(len(self.ids))

    def _read_from_zip(self, system_id):
        zip_path = os.path.join(SYSTEMS_DIR, f"{system_id[1:3]}.zip")
        if not os.path.exists(zip_path): raise FileNotFoundError(f"Zip bucket {zip_path} not found for {system_id}")
        with ZipFile(zip_path, 'r') as zf:
            with zf.open(f"{system_id}/receptor.pdb") as f:
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
            best_mol, max_atoms = None, -1
            for sdf in candidate_ligands:
                mol = Chem.MolFromMolBlock(sdf, sanitize=False)
                if mol is not None:
                    n_atoms = mol.GetNumAtoms()
                    if n_atoms > max_atoms:
                        max_atoms, best_mol = n_atoms, mol
            return system_id, receptor_pdb, best_mol

    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, item):
        actual_idx = self.valid_indices[item]
        item_id = self.ids[actual_idx]
        cached_data = self.cache.get(item_id)
        if cached_data is not None: return cached_data
        try:
            raw_system_id, receptor_pdb, ligand_mol = self._read_from_zip(item_id)
            if ligand_mol is None: return None
            mol_pos, mol_one_hot, mol_bonds = parse_molecule(ligand_mol)
            protein_pos, protein_one_hot, protein_contacts = parse_protein(receptor_pdb)
            if len(protein_pos) == 0: return None
            extra_ligands = []
            if self.random_ligand_smiles:
                sampled_smis = random.sample(self.random_ligand_smiles, min(50, len(self.random_ligand_smiles)))
                for smi in sampled_smis:
                    _, m_h_ex, m_b_ex = parse_smiles(smi)
                    if m_h_ex is not None: extra_ligands.append((m_h_ex, m_b_ex))
            result = build_graph(raw_system_id, raw_system_id, mol_pos, mol_one_hot, mol_bonds, protein_pos, protein_one_hot, protein_contacts, extra_ligands)
            if result is not None: self.cache.set(item_id, result)
            return result
        except: return None

def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)
