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
# from src.graph import Graph, batch as graph_batch
from io import StringIO
import pyarrow.parquet as pq
from zipfile import ZipFile
import logging

# Configure paths
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_ROOT, 'data', 'plinder', 'data', '2024-06', 'v2')
SYSTEMS_DIR = os.path.join(DATA_DIR, 'systems')
SPLIT_PATH = os.path.join(DATA_DIR, 'splits', 'split.parquet')
SAS_PATH = os.path.join(PROJ_ROOT, 'data', 'plinder', 'sas_points.zip')

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
    
    n_res = len(protein_pos)
    if n_res == 0:
         return np.array([]), np.array([]), np.zeros((0, 3)), np.zeros((0, 2))

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
        
    return protein_pos, protein_one_hot, protein_contacts, protein_backbone



def build_sample_features(protein_name, ligand_name, mol_pos, mol_h, mol_bonds, prot_pos, prot_h, prot_contacts, prot_backbone, sas_points=None, normalization_factor=1000.0):
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
    
    # 2. Coordinate Construction
    # Center coordinates on Protein Mean
    p_center = np.mean(prot_pos, axis=0, keepdims=True)
    prot_pos_centered = prot_pos - p_center + 500
    
    # Apply Positional Encoding to Protein Coordinates (Sin/Cos)
    # Treat each coordinate (x, y, z) as 'pos' in PE formula
    # Target dim per coord is 2 (sin, cos). So d_model=2, i=0. Denom = 10000^(0) = 1.
    # sin(pos/norm), cos(pos/norm)
    prot_x_sin = np.sin(prot_pos_centered / normalization_factor)
    prot_x_cos = np.cos(prot_pos_centered / normalization_factor)
    
    # Stack: [x_sin, y_sin, z_sin, x_cos, y_cos, z_cos] ? 
    # Or [x_sin, x_cos, y_sin, y_cos, z_sin, z_cos]?
    # Standard PE interleaves: [2i, 2i+1]
    # Let's stack as [N, 6] -> [sin_x, cos_x, sin_y, cos_y, sin_z, cos_z]
    prot_pos_enc = np.stack([
        prot_x_sin[:, 0], prot_x_cos[:, 0],
        prot_x_sin[:, 1], prot_x_cos[:, 1],
        prot_x_sin[:, 2], prot_x_cos[:, 2]
    ], axis=1) # [N, 6]
    
    # Protein H: [Original Features, Positional Encoding]
    prot_h_out = np.concatenate([prot_h, prot_pos_enc], axis=-1)
    
    # Ligand H: Original Features
    mol_h_out = mol_h 
    
    # SAS Points Processing
    sas_h_out = np.zeros((0, 6))
    if sas_points is not None and len(sas_points) > 0:
        # Center SAS points
        sas_pos_centered = sas_points - p_center + 500
        # Apply Positional Encoding
        sas_x_sin = np.sin(sas_pos_centered / normalization_factor)
        sas_x_cos = np.cos(sas_pos_centered / normalization_factor)
        sas_h_out = np.stack([
            sas_x_sin[:, 0], sas_x_cos[:, 0],
            sas_x_sin[:, 1], sas_x_cos[:, 1],
            sas_x_sin[:, 2], sas_x_cos[:, 2]
        ], axis=1) # [N_sas, 6]

    # 4. Pocket Identification (SAS points based)
    # Find SAS point closest to ligand center -> is_pocket
    # Random other SAS point -> is_decoy
    n_sas = len(sas_points)
    is_pocket = np.zeros(n_sas, dtype=np.float32)
    is_decoy = np.zeros(n_sas, dtype=np.float32)

    if n_sas > 0 and len(mol_pos) > 0:
        mol_center = np.mean(mol_pos, axis=0)
        # sas_points are raw coordinates passed in, same space as mol_pos
        dists = np.linalg.norm(sas_points - mol_center, axis=1)
        closest_idx = np.argmin(dists)
        is_pocket[closest_idx] = 1.0
        
        # Select Decoy
        # Choose any index except closest_idx
        # If n_sas == 1, we can't choose a different decoy.
        if n_sas > 1:
            possible_indices = np.arange(n_sas)
            possible_indices = np.delete(possible_indices, closest_idx)
            decoy_idx = np.random.choice(possible_indices)
            is_decoy[decoy_idx] = 1.0

    return {
        'prot': torch.tensor(prot_h_out, dtype=const.TORCH_FLOAT),
        'mol': torch.tensor(mol_h_out, dtype=const.TORCH_FLOAT),
        'sas': torch.tensor(sas_h_out, dtype=const.TORCH_FLOAT),
        'is_pocket': torch.tensor(is_pocket, dtype=const.TORCH_FLOAT),
        'is_decoy': torch.tensor(is_decoy, dtype=const.TORCH_FLOAT),
        'pname': p_name,
        'lname': l_name
    }

def collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    max_p = max([b['prot'].shape[0] for b in batch])
    max_m = max([b['mol'].shape[0] for b in batch])
    max_s = max([b['sas'].shape[0] for b in batch]) if batch else 0
    B = len(batch)
    
    # Global sequence length for Protein + Ligand
    max_pm = max_p + max_m
    
    prot_dim = batch[0]['prot'].shape[1]
    mol_dim = batch[0]['mol'].shape[1]
    sas_dim = 6 
    
    # Feature Tensors (Separate, padded to their own max lengths)
    prot_out = torch.zeros((B, max_p, prot_dim), dtype=const.TORCH_FLOAT)
    mol_out = torch.zeros((B, max_m, mol_dim), dtype=const.TORCH_FLOAT)
    sas_out = torch.zeros((B, max_s, sas_dim), dtype=const.TORCH_FLOAT)
    
    # Labels aligned with SAS
    is_pocket_out = torch.zeros((B, max_s), dtype=const.TORCH_FLOAT)
    is_decoy_out = torch.zeros((B, max_s), dtype=const.TORCH_FLOAT)
    
    # Masks
    # prot_mask and mol_mask are on the concatenated [Prot, Mol] sequence
    prot_mask = torch.zeros((B, max_pm), dtype=torch.bool)
    mol_mask  = torch.zeros((B, max_pm), dtype=torch.bool)
    
    # sas_mask for the sas tensor itself
    sas_mask  = torch.zeros((B, max_s), dtype=torch.bool)
    
    pnames = []
    lnames = []
    
    for i, b in enumerate(batch):
        p_len = b['prot'].shape[0]
        m_len = b['mol'].shape[0]
        s_len = b['sas'].shape[0]
        
        # Features
        prot_out[i, :p_len] = b['prot']
        mol_out[i, :m_len]  = b['mol']
        if s_len > 0:
            sas_out[i, :s_len] = b['sas']
            is_pocket_out[i, :s_len] = b['is_pocket']
            is_decoy_out[i, :s_len] = b['is_decoy']
            sas_mask[i, :s_len] = True
            
        # Masks on [Prot, Mol] Grid
        # 1. Protein Segment: [0, p_len]
        prot_mask[i, :p_len] = True
        
        # 2. Mol Segment: [max_p, max_p + m_len]
        mol_start = max_p
        mol_mask[i, mol_start : mol_start + m_len] = True
        
        pnames.append(b.get('pname', ''))
        lnames.append(b.get('lname', ''))
        
    return {
        'prot': prot_out,
        'prot_mask': prot_mask, # [B, max_p + max_m]
        'mol': mol_out,
        'mol_mask': mol_mask,   # [B, max_p + max_m]
        'sas': sas_out,
        'sas_mask': sas_mask,   # [B, max_s]
        'is_pocket': is_pocket_out, # [B, max_s]
        'is_decoy': is_decoy_out,   # [B, max_s]
        'pname': pnames,
        'lname': lnames
    }

class PocketDataset(Dataset):
    collate_fn = staticmethod(collate)
    def __init__(self, device=None, progress_bar=True, split='train', limit=None):
        self.progress_bar = progress_bar
        self.device = device
        self.split = split
        # Use file-based cache for persistence
        self.cache = FileCache(cache_mode=None, dataset_name=f'pocket_dataset_{split}')

        # Determine split filter
        split_filter = split
        if split == 'validation':
            split_filter = 'val'
            
        from src.positions2.config import get_config
        config = get_config()
        self.normalization_factor = config.get('normalization_factor', 1000.0)
        
        # Check SAS file existence
        if not os.path.exists(SAS_PATH):
             print(f"Warning: SAS file not found at {SAS_PATH}")
        
        self.sas_zip = None # Lazy loading in getitem

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
        
        # Check if item is in cache
        cached_data = self.cache.get(item_id)
        if cached_data is not None:
             return cached_data
        
        # Get the sample data from ZIP directly
        system_id = item_id 
        
        raw_system_id, receptor_pdb, ligand_mol = self._read_from_zip(system_id)
        
        if ligand_mol is None: # Should be caught by read_from_zip but safe to check
             raise ValueError(f"Ligand mol is None for {system_id}")

        mol_pos, mol_one_hot, mol_bonds = parse_molecule(ligand_mol)
        protein_pos, protein_one_hot, protein_contacts, protein_backbone = parse_protein(receptor_pdb)
        
        # Retrieve SAS points from ZIP (Lazy Load)
        if self.sas_zip is None:
            if os.path.exists(SAS_PATH):
                self.sas_zip = ZipFile(SAS_PATH, 'r')
            else:
                self.sas_zip = None

        sas_points = np.zeros((0, 3))
        if self.sas_zip is not None:
            try:
                with self.sas_zip.open(f"{system_id}.npy") as f:
                    sas_points = np.load(f)
            except KeyError:
                pass # Not found for this system
            except Exception as e:
                print(f"Error reading SAS for {system_id}: {e}")

        if len(protein_pos) == 0:
              raise ValueError(f"No valid residues found for {system_id}")

        result = build_sample_features(
            protein_name=raw_system_id,
            ligand_name=raw_system_id,
            mol_pos=mol_pos,
            mol_h=mol_one_hot,
            mol_bonds=mol_bonds,
            prot_pos=protein_pos,
            prot_h=protein_one_hot,
            prot_contacts=protein_contacts,
            prot_backbone=protein_backbone,
            sas_points=sas_points,
            normalization_factor=self.normalization_factor
        )
        
        if result is None:
            return None
            
        self.cache.set(item_id, result)
        return result



def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

