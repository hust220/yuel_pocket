import pickle
import torch

# Load numpy arrays from bytes using pickle
molecule_pos = pickle.loads(molecule_pos_bytes)
molecule_one_hot = pickle.loads(molecule_one_hot_bytes)
molecule_bonds = pickle.loads(molecule_bonds_bytes)
protein_pos = pickle.loads(protein_pos_bytes)
protein_one_hot = pickle.loads(protein_one_hot_bytes)
protein_contacts = pickle.loads(protein_contacts_bytes)
protein_backbone = pickle.loads(protein_backbone_bytes)
is_pocket = pickle.loads(is_pocket_bytes)

# Convert to torch tensors
molecule_pos = torch.from_numpy(molecule_pos).float()
molecule_one_hot = torch.from_numpy(molecule_one_hot).float()
molecule_bonds = torch.from_numpy(molecule_bonds).float()
protein_pos = torch.from_numpy(protein_pos).float()
protein_one_hot = torch.from_numpy(protein_one_hot).float()
protein_contacts = torch.from_numpy(protein_contacts).float()
protein_backbone = torch.from_numpy(protein_backbone).long()
is_pocket = torch.from_numpy(is_pocket).bool() 