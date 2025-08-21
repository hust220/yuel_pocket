import torch

from rdkit import Chem

def generate_mappings(items_list):
    item2idx = {item: idx for idx, item in enumerate(items_list)}
    idx2item = {idx: item for idx, item in enumerate(items_list)}
    
    return item2idx, idx2item

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int32

ALLOWED_ATOM_TYPES = [
    'C', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I', 'Cu', 'Se', 'Fe', 'V', 'Mo', 'B', 'Co', 
    'X' # unknown atom type
]
ALLOWED_RESIDUE_TYPES = [
    # Standard amino acids
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
    
    # Modified amino acids (less common)
    'SEC', 'PYL', 'SEP', 'TPO', 'PTR',
    
    # Nucleotides (if working with DNA/RNA interfaces)
    'A', 'C', 'G', 'T', 'U', 'DA', 'DC', 'DG', 'DT',
    
    # Common cofactors/metals
    'HEM', 'FAD', 'NAD', 'ATP', 'GTP', 

    'UNK' # unknown residue type
]

RDKIT_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.DATIVE,
    Chem.rdchem.BondType.IONIC,
    Chem.rdchem.BondType.HYDROGEN,
    Chem.rdchem.BondType.THREECENTER,
    Chem.rdchem.BondType.UNSPECIFIED,
    Chem.rdchem.BondType.ZERO # No Bond
]
BOND_TYPE_NAMES = ['Single', 'Double', 'Triple', 'Aromatic', 'Dative', 'Ionic', 'Hydrogen', 'ThreeCenter', 'Unspecified', 'No Bond']

ATOM2IDX, IDX2ATOM = generate_mappings(ALLOWED_ATOM_TYPES)
RESIDUE2IDX, IDX2RESIDUE = generate_mappings(ALLOWED_RESIDUE_TYPES)
RDBOND2IDX, IDX2RDBOND = generate_mappings(RDKIT_BOND_TYPES)
N_ATOM_TYPES = len(ATOM2IDX)
N_RESIDUE_TYPES = len(RESIDUE2IDX)
N_RDBOND_TYPES = len(RDBOND2IDX)
