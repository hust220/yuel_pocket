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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import const
from Bio.PDB import PDBParser
from src.db_utils import db_connection

def parse_residues(rs):
    pocket_coords = []
    pocket_types = []

    for residue in rs:
        residue_name = residue.get_resname()
        
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            atom_coord = atom.get_coord()

            if atom_name == 'CA':
                pocket_coords.append(atom_coord.tolist())
                pocket_types.append(residue_name)

    return {
        'coord': pocket_coords,
        'types': pocket_types,
    }

def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule

# one hot for atoms
def atom_one_hot(atom):
    n = const.N_ATOM_TYPES
    one_hot = np.zeros(n)
    if atom not in const.ATOM2IDX:
        atom = 'X'
    one_hot[const.ATOM2IDX[atom]] = 1
    return one_hot

# one hot for amino acids
def aa_one_hot(residue):
    n = const.N_RESIDUE_TYPES
    one_hot = np.zeros(n)
    if residue not in const.RESIDUE2IDX:
        residue = 'UNK'
    one_hot[const.RESIDUE2IDX[residue]] = 1
    return one_hot

def bond_one_hot(bond):
    one_hot = [0 for i in range(const.N_RDBOND_TYPES)]
    
    # Set the appropriate index to 1
    bond_type = bond.GetBondType()
    if bond_type not in const.RDBOND2IDX:
        bond_type = Chem.rdchem.BondType.ZERO
    one_hot[const.RDBOND2IDX[bond_type]] = 1
        
    return one_hot

def parse_molecule(mol):
    atom_one_hots = []
    non_h_indices = []  # Keep track of non-hydrogen atom indices
    
    # First pass: collect non-hydrogen atoms and their indices
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() != 'H':
            atom_one_hots.append(atom_one_hot(atom.GetSymbol()))
            non_h_indices.append(idx)

    # Get positions for non-hydrogen atoms
    if mol.GetNumConformers() == 0:
        positions = np.zeros((len(non_h_indices), 3))
    else:
        all_positions = mol.GetConformer().GetPositions()
        positions = all_positions[non_h_indices]

    # Get bonds between non-hydrogen atoms
    bonds = []
    old_idx_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(non_h_indices)}
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Only include bonds where both atoms are non-hydrogen
        if i in old_idx_to_new and j in old_idx_to_new:
            one_hot = bond_one_hot(bond)
            new_i = old_idx_to_new[i]
            new_j = old_idx_to_new[j]
            bonds.append([new_i, new_j] + one_hot)
            bonds.append([new_j, new_i] + one_hot)

    return positions, np.array(atom_one_hots), np.array(bonds)

def parse_pocket(rs):
    pocket_coords = []
    pocket_types = []

    for residue in rs:
        residue_name = residue.get_resname()
        
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            atom_coord = atom.get_coord()

            if atom_name == 'CA':
                pocket_coords.append(atom_coord.tolist())
                pocket_types.append(residue_name)

    pocket_one_hot = []
    for _type in pocket_types:
        pocket_one_hot.append(aa_one_hot(_type))
    pocket_one_hot = np.array(pocket_one_hot)

    return pocket_coords, pocket_one_hot

def get_pocket(mol, pdb_path):
    struct = PDBParser().get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []

    for ir,residue in enumerate(struct.get_residues()):
        for atom in residue.get_atoms():
            atom_coords.append(atom.get_coord())
            residue_ids.append(ir)

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    mol_atom_coords = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    contact_residues = np.unique(residue_ids[np.where(distances.min(axis=1) <= 6)[0]])

    return parse_pocket([r for (ir, r) in enumerate(struct.get_residues()) if ir in contact_residues])

def pad_and_concatenate(tensor1, tensor2):
    N, a = tensor1.shape
    M, b = tensor2.shape
    
    # Pad tensor1 with zeros for the b columns it's missing
    tensor1_padded = np.pad(tensor1, 
                           pad_width=((0, 0), (0, b)),  # Pad b zeros on the right
                           mode='constant',
                           constant_values=0)
    
    # Pad tensor2 with zeros for the a columns it's missing
    tensor2_padded = np.pad(tensor2,
                           pad_width=((0, 0), (a, 0)),  # Pad a zeros on the left
                           mode='constant',
                           constant_values=0)
    
    # Concatenate along the first axis (stack vertically)
    return np.concatenate([tensor1_padded, tensor2_padded], axis=0)

def get_edges(protein_contacts, protein_backbone, mol_bonds, protein_size, mol_size):
    # protein_contacts: n_contacts 3
    # protein_backbone: n_neighbors 2
    # mol_bonds: n_bonds 2+bond_features

    # edge features: distance, backbone neighbor, protein-joint, joint-mol, bond_one_hot
    # n_edge_features = 1 + 1 + 1 + 1 + n_bond_feats

    n_contacts = protein_contacts.shape[0]
    n_neighbors = protein_backbone.shape[0]
    n_bonds = mol_bonds.shape[0]
    n_bond_feats = mol_bonds.shape[1] - 2

    n_edge_features = 1 + 1 + 1 + 1 + n_bond_feats

    edge_index = []
    edge_attr = []
    bond_orders = []

    # protein - protein (contacts)
    for contact in protein_contacts:
        edge_index.append([contact[0]-1, contact[1]-1])
        edge_attr.append([contact[2]] + [0] * (n_edge_features - 1))

    # protein - protein (backbone)
    for neighbor in protein_backbone:
        # if (neighbor[0], neighbor[1]) is in edge_index, find the index and replace the edge_attr
        if [neighbor[0], neighbor[1]] in edge_index:
            idx = edge_index.index([neighbor[0], neighbor[1]])
            edge_attr[idx][1] = 1
        elif [neighbor[1], neighbor[0]] in edge_index:
            idx = edge_index.index([neighbor[1], neighbor[0]])
            edge_attr[idx][1] = 1
        else:
            edge_index.append([neighbor[0], neighbor[1]])
            edge_attr.append([0,1] + [0] * (n_edge_features - 2))

    # protein - joint
    for i in range(protein_size):
        edge_index.append([i, protein_size])
        edge_attr.append([0,0,1] + [0] * (n_edge_features - 3))

    # joint - compound
    for i in range(mol_size):
        edge_index.append([protein_size, i+protein_size+1])
        edge_attr.append([0,0,0,1] + [0] * (n_edge_features - 4))

    # compound - compound
    for bond in mol_bonds:
        # print('bond: ', bond)
        edge_index.append([bond[0]+protein_size+1, bond[1]+protein_size+1])
        edge_attr.append([0,0,0,0] + bond[2:].tolist())
    
    edge_mask = np.ones(len(edge_index))

    return edge_index, edge_attr, edge_mask

def process_single_item(row, device):
    """Process a single data item"""
    try:
        ligand_name = row['ligand_name']
        protein_name = row['protein_name']

        mol_pos = row['molecule_pos'] # n_atoms 3
        mol_one_hot = row['molecule_one_hot'] # n_atoms mol_node_features
        mol_bonds = row['molecule_bonds'] # n_bonds 2+bond_features
        protein_pos = row['protein_pos'] # n_atoms 3
        protein_one_hot = row['protein_one_hot'] # n_atoms protein_node_features
        protein_contacts = row['protein_contacts'] # n_contacts 3
        protein_backbone = row['protein_backbone'] # n_neighbors 2

        # expand the last dimension of mol_one_hot and protein_one_hot to the same length
        mol_size, mol_node_features = mol_one_hot.shape
        protein_size, protein_node_features = protein_one_hot.shape

        if protein_size > 1000:
            return None

        protein_one_hot = np.concatenate([protein_one_hot, np.zeros((protein_size, mol_node_features+1))], axis=-1)
        joint_one_hot = np.concatenate([np.zeros((1, protein_node_features)), np.ones((1, 1)), np.zeros((1, mol_node_features))], axis=-1)
        mol_one_hot = np.concatenate([np.zeros((mol_size, protein_node_features+1)), mol_one_hot], axis=-1)
        one_hot = np.concatenate([protein_one_hot, joint_one_hot, mol_one_hot], axis=0)

        protein_mask = np.zeros(protein_size + 1 + mol_size)
        protein_mask[:protein_size] = 1

        node_mask = np.ones(protein_size + 1 + mol_size)

        edge_index, edge_attr, edge_mask = get_edges(protein_contacts, protein_backbone, mol_bonds, protein_size, mol_size)

        is_pocket = np.zeros(protein_size + 1 + mol_size)
        is_pocket[:protein_size] = row['is_pocket']

        # Convert numpy arrays to lists before creating tensors
        return {
            'pname': protein_name,
            'lname': ligand_name,
            'one_hot': one_hot.tolist(),  # Convert to list
            'edge_index': np.array(edge_index).tolist(),  # Convert to list
            'edge_attr': np.array(edge_attr).tolist(),  # Convert to list
            'node_mask': node_mask.tolist(),  # Convert to list
            'edge_mask': np.array(edge_mask).tolist(),  # Convert to list
            'protein_mask': protein_mask.tolist(),  # Convert to list
            'is_pocket': is_pocket.tolist(),  # Convert to list
        }
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        print(f"Protein name: {row.get('protein_name', 'unknown')}")
        print(f"Ligand name: {row.get('ligand_name', 'unknown')}")
        raise e
        return None

def process_chunk(chunk_ids):
    """Process a chunk of data and save directly to database"""
    processed_data = []
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            # Fetch raw data for the given IDs
            cursor.execute("""
                SELECT id, ligand_name, protein_name, molecule_pos, molecule_one_hot,
                       molecule_bonds, protein_pos, protein_one_hot, protein_contacts,
                       protein_backbone, is_pocket
                FROM raw_datasets
                WHERE id = ANY(%s)
            """, (chunk_ids,))
            raw_items = cursor.fetchall()
            cursor.close()

        for raw_item in raw_items:
            try:
                raw_dict = {
                    'ligand_name': raw_item[1],
                    'protein_name': raw_item[2],
                    'molecule_pos': pickle.loads(raw_item[3]),
                    'molecule_one_hot': pickle.loads(raw_item[4]),
                    'molecule_bonds': pickle.loads(raw_item[5]),
                    'protein_pos': pickle.loads(raw_item[6]),
                    'protein_one_hot': pickle.loads(raw_item[7]),
                    'protein_contacts': pickle.loads(raw_item[8]),
                    'protein_backbone': pickle.loads(raw_item[9]),
                    'is_pocket': pickle.loads(raw_item[10]),
                }

                result = process_single_item(raw_dict, torch.device('cpu'))
                if result is not None:
                    # Convert all data to bytes
                    processed_data.append((
                        result['pname'],
                        result['lname'],
                        pickle.dumps(np.array(result['one_hot'], dtype=np.float32)),
                        pickle.dumps(np.array(result['edge_index'], dtype=np.int64)),
                        pickle.dumps(np.array(result['edge_attr'], dtype=np.float32)),
                        pickle.dumps(np.array(result['node_mask'], dtype=np.float32)),
                        pickle.dumps(np.array(result['edge_mask'], dtype=np.float32)),
                        pickle.dumps(np.array(result['protein_mask'], dtype=np.float32)),
                        pickle.dumps(np.array(result['is_pocket'], dtype=np.float32))
                    ))
            except Exception as e:
                print(f"Error processing item {raw_item[0]}: {str(e)}")
                continue

        # Save processed data to database
        if processed_data:
            try:
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT INTO processed_datasets (
                            pname, lname, one_hot, edge_index, edge_attr,
                            node_mask, edge_mask, protein_mask, is_pocket
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (pname, lname) DO NOTHING
                    """, processed_data)
                    conn.commit()
                    cursor.close()
            except Exception as e:
                print(f"Error saving to database: {str(e)}")

    except Exception as e:
        print(f"Error in process_chunk: {str(e)}")

    return len(processed_data)  # Return the number of successfully processed items

def parallel_preprocess(num_workers=4, batch_size=10):
    """Process data in parallel using multiprocessing Pool"""
    # Create the processed_datasets table
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_datasets (
                    id SERIAL PRIMARY KEY,
                    pname TEXT,
                    lname TEXT,
                    one_hot BYTEA,
                    edge_index BYTEA,
                    edge_attr BYTEA,
                    node_mask BYTEA,
                    edge_mask BYTEA,
                    protein_mask BYTEA,
                    is_pocket BYTEA,
                    UNIQUE(pname, lname)
                )
            """)
            conn.commit()
            cursor.close()
    except Exception as e:
        print(f"Error creating database table: {str(e)}")

    # Get all IDs from raw_datasets
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM raw_datasets ORDER BY id")
            all_ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
    except Exception as e:
        print(f"Error fetching IDs from raw_datasets: {str(e)}")

    # Split IDs into chunks for parallel processing
    chunks = [all_ids[i:i + batch_size] for i in range(0, len(all_ids), batch_size)]

    # Process chunks in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="Processing data chunks"
        ))

    # Print summary
    total_processed = sum(results)
    print(f"Successfully processed {total_processed} items out of {len(all_ids)} total items")

    return None  # Data is already saved to database

class PocketDataset(Dataset):
    # Class variables for data attributes
    DATA_LIST_ATTRS = ['pname', 'lname']  # List attributes that don't need padding
    DATA_ATTRS_TO_PAD = ['one_hot', 'edge_index', 'edge_attr', 'node_mask', 'edge_mask', 'protein_mask', 'is_pocket']  # Attributes that need padding
    DATA_ATTRS_TO_ADD_LAST_DIM = ['node_mask', 'edge_mask', 'protein_mask', 'is_pocket']  # Attributes that need an extra dimension

    def __init__(self, data=None, device=None, progress_bar=True, split='train'):
        self.progress_bar = progress_bar
        self.device = device
        self.split = split
        self.cache = {}  # Instance-level cache

        if data is not None:
            self.data = data
            self.valid_indices = np.arange(len(data))
            return
        else:
            try:
                with db_connection() as conn:
                    start_time = time.time()
                    print('Loading dataset IDs from database...')
                    cursor = conn.cursor()                    
                    cursor.execute("""
                        SELECT id
                        FROM processed_datasets
                        WHERE split = %s
                        ORDER BY id
                    """, (split,))
                    self.ids = [row[0] for row in cursor.fetchall()]
                    cursor.close()
                    print('Loaded dataset IDs from database, Time: ', time.time() - start_time, 's')
                
                self.valid_indices = np.arange(len(self.ids))
                print(f"Found {len(self.valid_indices)} valid entries")
            except Exception as e:
                print(f"Error loading dataset from database: {str(e)}")
                raise e

    def __len__(self):
        return len(self.valid_indices)

    def _create_tensors(self, data):
        """Helper method to create tensors from data dictionary."""
        pname, lname, one_hot, edge_index, edge_attr, node_mask, edge_mask, protein_mask, is_pocket = data
        return {
            'pname': pname,
            'lname': lname,
            'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=self.device),
            'edge_index': torch.tensor(edge_index, dtype=torch.long, device=self.device),
            'edge_attr': torch.tensor(edge_attr, dtype=const.TORCH_FLOAT, device=self.device),
            'node_mask': torch.tensor(node_mask, dtype=const.TORCH_INT, device=self.device),
            'edge_mask': torch.tensor(edge_mask, dtype=const.TORCH_INT, device=self.device),
            'protein_mask': torch.tensor(protein_mask, dtype=const.TORCH_INT, device=self.device),
            'is_pocket': torch.tensor(is_pocket, dtype=const.TORCH_FLOAT, device=self.device)
        }

    def __getitem__(self, item):
        # Map the requested index to the actual index in the dataset
        actual_idx = self.valid_indices[item]
        item_id = self.ids[actual_idx]
        
        # Check if item is in cache
        if item_id in self.cache:
            return self._create_tensors(self.cache[item_id])
        
        # Get the sample data from database
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, pname, lname, one_hot, edge_index, edge_attr,
                       node_mask, edge_mask, protein_mask, is_pocket
                FROM processed_datasets
                WHERE id = %s
            """, (item_id,))
            sample = cursor.fetchone()
            cursor.close()
                
        # Cache the unpickled data
        self.cache[item_id] = (
            sample[1],
            sample[2],
            pickle.loads(sample[3]),
            pickle.loads(sample[4]),
            pickle.loads(sample[5]),
            pickle.loads(sample[6]),
            pickle.loads(sample[7]),
            pickle.loads(sample[8]),
            pickle.loads(sample[9])
        )
        
        # Convert to tensors
        return self._create_tensors(self.cache[item_id])

def collate(batch):
    out = {}

    # collect the list attributes
    for data in batch:
        for key, value in data.items():
            if key in PocketDataset.DATA_LIST_ATTRS or key in PocketDataset.DATA_ATTRS_TO_PAD:
                out.setdefault(key, []).append(value)

    # pad the tensors
    for key, value in out.items():
        if key in PocketDataset.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue

    # add last dimension to the tensor
    for key in PocketDataset.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

def test_pocket_dataset():
    """Test function for PocketDataset class, collate function and get_dataloader"""
    print("\nTesting PocketDataset initialization...")
    try:
        # Initialize dataset
        dataset = PocketDataset(device=torch.device('cpu'))
        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) > 0, "Dataset should not be empty"
        
        # Test single item access
        print("\nTesting single item access...")
        sample = dataset[0]
        pname = sample['pname']
        lname = sample['lname']
        
        # Get contacts from proteins table
        print(f"\nFetching contacts for protein {pname} from proteins table...")
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT contacts 
                FROM proteins 
                WHERE name = %s
            """, (pname,))
            result = cursor.fetchone()
            cursor.close()
            
            if result and result[0] is not None:
                contacts_data = result[0]
                contents = contacts_data.tobytes().decode('utf-8')
                print("\nRaw contacts data:")
                print(contents[:1000])  # Print first 1000 characters
                
                # Parse contacts
                contacts = []
                for line in contents.split('\n'):
                    if line:
                        parts = line.split()
                        contacts.append([int(parts[0]), int(parts[1]), float(parts[2])])
                contacts = np.array(contacts)
                print("\nParsed contacts (first 20 rows):")
                print(contacts[:20])
                print(f"Total number of contacts: {len(contacts)}")
                print(f"Contacts dtype: {contacts.dtype}")
                print(f"Contacts shape: {contacts.shape}")
            else:
                print(f"No contacts found for protein {pname}")
        
        # Get raw data from database for verification
        print(f"\nVerifying edge generation for protein {pname} and ligand {lname}...")
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT molecule_bonds, protein_contacts, protein_backbone, protein_one_hot
                FROM raw_datasets
                WHERE protein_name = %s AND ligand_name = %s
            """, (pname, lname))
            raw_data = cursor.fetchone()
            cursor.close()
        
        if raw_data:
            # Convert bytes back to numpy arrays using pickle
            molecule_bonds = pickle.loads(raw_data[0])
            protein_contacts = pickle.loads(raw_data[1])
            protein_backbone = pickle.loads(raw_data[2])
            protein_one_hot = pickle.loads(raw_data[3])
            
            # Get dimensions
            protein_size = protein_one_hot.shape[0]
            mol_size = int((sample['one_hot'].shape[0] - protein_size - 1) / 2)  # -1 for joint node, /2 for symmetry
            
            print(f"Protein size: {protein_size}")
            print(f"Molecule size: {mol_size}")
            
            # Generate edges using get_edges
            edge_index, edge_attr, edge_mask = get_edges(
                protein_contacts, protein_backbone, molecule_bonds,
                protein_size, mol_size
            )
            
            # Convert to numpy arrays for comparison
            edge_index = np.array(edge_index)
            edge_attr = np.array(edge_attr)
            
            # Compare with processed data
            print("\nComparing edge generation results:")
            print(f"Original edge_index shape: {edge_index.shape}")
            print(f"Processed edge_index shape: {sample['edge_index'].shape}")
            print(f"Original edge_attr shape: {edge_attr.shape}")
            print(f"Processed edge_attr shape: {sample['edge_attr'].shape}")
            
            # Print some sample edges for comparison
            print("\nSample edges from original generation:")
            print(edge_index[-10:])
            print("\nSample edges from processed data:")
            print(sample['edge_index'][-10:].cpu().numpy())
            
            # Verify edge attributes
            print("\nSample edge attributes from original generation:")
            print(edge_attr[-10:])
            print("\nSample edge attributes from processed data:")
            print(sample['edge_attr'][-10:].cpu().numpy())
        
        # Continue with existing tests...
        required_fields = [
            'pname', 'lname', 'one_hot', 'edge_index', 'edge_attr',
            'node_mask', 'edge_mask', 'protein_mask', 'is_pocket'
        ]
        for field in required_fields:
            assert field in sample, f"Sample should contain {field}"
            if field in ['pname', 'lname']:
                print(f"{field}: {sample[field]}")
            else:
                print(f"{field} shape: {sample[field].shape}")
        
        # Test tensor types and devices
        print("\nTesting tensor types and devices...")
        assert sample['one_hot'].dtype == const.TORCH_FLOAT, "one_hot should be float tensor"
        assert sample['edge_index'].dtype == torch.long, "edge_index should be long tensor"
        assert sample['edge_attr'].dtype == const.TORCH_FLOAT, "edge_attr should be float tensor"
        assert sample['node_mask'].dtype == const.TORCH_INT, "node_mask should be int tensor"
        assert sample['edge_mask'].dtype == const.TORCH_INT, "edge_mask should be int tensor"
        assert sample['protein_mask'].dtype == const.TORCH_INT, "protein_mask should be int tensor"
        assert sample['is_pocket'].dtype == const.TORCH_FLOAT, "is_pocket should be float tensor"
        
        # Test tensor shapes
        print("\nTesting tensor shapes...")
        assert sample['one_hot'].shape[0] > 0, "One-hot tensor should have nodes"
        assert sample['edge_index'].shape[1] == 2, "Edge index tensor should have 2 columns"
        assert sample['edge_attr'].shape[1] > 0, "Edge attribute tensor should have features"
        
        # Test mask values
        print("\nTesting mask values...")
        assert torch.all(sample['node_mask'] >= 0) and torch.all(sample['node_mask'] <= 1), "node_mask should be binary"
        assert torch.all(sample['edge_mask'] >= 0) and torch.all(sample['edge_mask'] <= 1), "edge_mask should be binary"
        assert torch.all(sample['protein_mask'] >= 0) and torch.all(sample['protein_mask'] <= 1), "protein_mask should be binary"
        assert torch.all(sample['is_pocket'] >= 0) and torch.all(sample['is_pocket'] <= 1), "is_pocket should be binary"
        
        # Test data consistency
        print("\nTesting data consistency...")
        n_nodes = sample['one_hot'].shape[0]
        assert sample['node_mask'].shape[0] == n_nodes, "node_mask should match number of nodes"
        assert sample['protein_mask'].shape[0] == n_nodes, "protein_mask should match number of nodes"
        assert sample['is_pocket'].shape[0] == n_nodes, "is_pocket should match number of nodes"
        
        n_edges = sample['edge_index'].shape[0]
        assert sample['edge_attr'].shape[0] == n_edges, "edge_attr should match number of edges"
        assert sample['edge_mask'].shape[0] == n_edges, "edge_mask should match number of edges"
        
        # Test edge indices
        print("\nTesting edge indices...")
        max_node_idx = n_nodes - 1
        edge_index = sample['edge_index']
        n_protein_nodes = torch.sum(sample['protein_mask'])
        print('n_nodes: ', n_nodes)
        print('n_protein_nodes: ', n_protein_nodes)
        print(edge_index[:20])
        print(edge_index[-20:])
        invalid_indices = torch.where((edge_index < 0) | (edge_index > max_node_idx))
        if len(invalid_indices[0]) > 0:
            print(f"Found {len(invalid_indices[0])} invalid edge indices:")
            print(f"Edge index shape: {edge_index.shape}")
            print(f"Max node index: {max_node_idx}")
            print(f"Invalid indices positions: {invalid_indices}")
            print(f"Invalid values: {edge_index[invalid_indices]}")
            print(edge_index)
            raise AssertionError("Edge indices should be within valid node range")
        print("All edge indices are valid")
        
        # Test collate function
        print("\nTesting collate function...")
        batch = [dataset[i] for i in range(min(3, len(dataset)))]  # Get a small batch
        collated = collate(batch)
        print("Collated batch shapes:")
        for key, value in collated.items():
            if key in ['pname', 'lname']:
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {value.shape}")
        
        # Test dataloader
        print("\nTesting dataloader...")
        batch_size = 2
        dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
        batch = next(iter(dataloader))
        print("Dataloader batch shapes:")
        for key, value in batch.items():
            if key in ['pname', 'lname']:
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {value.shape}")
        print('batch[one_hot].dtype', batch['one_hot'].dtype)
        print('batch[edge_index].dtype', batch['edge_index'].dtype)
        print('batch[edge_attr].dtype', batch['edge_attr'].dtype)
        print('batch[node_mask].dtype', batch['node_mask'].dtype)
        print('batch[edge_mask].dtype', batch['edge_mask'].dtype)
        print('batch[protein_mask].dtype', batch['protein_mask'].dtype)
        print('batch[is_pocket].dtype', batch['is_pocket'].dtype)
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        raise e
    finally:
        # Ensure the dataset is properly closed
        if 'dataset' in locals():
            del dataset

def set_split():
    """Add a split column to processed_datasets and assign data to train/val/test sets.
    Proteins from COACH420 and Holo4K will be assigned to test set.
    Others will be split between train and validation sets (50 for validation)."""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            
            # Add split column if it doesn't exist
            cursor.execute("""
                ALTER TABLE processed_datasets 
                ADD COLUMN IF NOT EXISTS split TEXT
            """)
            
            # Print initial statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN in_coach420 IS TRUE THEN 1 ELSE 0 END) as coach420_count,
                    SUM(CASE WHEN in_holo4k IS TRUE THEN 1 ELSE 0 END) as holo4k_count,
                    SUM(CASE WHEN in_coach420 IS TRUE AND in_holo4k IS TRUE THEN 1 ELSE 0 END) as both_count
                FROM processed_datasets
            """)
            initial_stats = cursor.fetchone()
            print("\nInitial dataset statistics:")
            print(f"Total proteins: {initial_stats[0]}")
            print(f"Proteins in COACH420: {initial_stats[1]}")
            print(f"Proteins in Holo4K: {initial_stats[2]}")
            print(f"Proteins in both datasets: {initial_stats[3]}")
            
            # First, assign test set (COACH420 and Holo4K proteins)
            cursor.execute("""
                UPDATE processed_datasets 
                SET split = 'test' 
                WHERE in_coach420 IS TRUE OR in_holo4k IS TRUE
            """)
            
            # Get remaining proteins (not in test set) for train/val split
            cursor.execute("""
                SELECT id 
                FROM processed_datasets 
                WHERE in_coach420 IS FALSE AND in_holo4k IS FALSE
                ORDER BY id
            """)
            remaining_ids = [row[0] for row in cursor.fetchall()]
            
            print(f"\nNumber of remaining proteins for train/val split: {len(remaining_ids)}")
            
            if len(remaining_ids) == 0:
                print("\nWarning: No proteins remaining for train/val split!")
                print("All proteins are from COACH420 or Holo4K.")
                return
            
            # Randomly select 50 for validation (or all if less than 50)
            np.random.seed(42)  # For reproducibility
            val_size = min(50, len(remaining_ids))
            val_indices = np.random.choice(len(remaining_ids), size=val_size, replace=False)
            val_ids = [remaining_ids[i] for i in val_indices]
            
            # Update validation set
            cursor.execute("""
                UPDATE processed_datasets 
                SET split = 'val' 
                WHERE id = ANY(%s)
            """, (val_ids,))
            
            # Update remaining as training set
            cursor.execute("""
                UPDATE processed_datasets 
                SET split = 'train' 
                WHERE split IS NULL
            """)
            
            # Verify the split
            cursor.execute("""
                SELECT split, COUNT(*) 
                FROM processed_datasets 
                GROUP BY split
            """)
            split_counts = cursor.fetchall()
            
            print("\nSplit distribution:")
            for split, count in split_counts:
                print(f"{split}: {count} samples")
            
            # Print additional statistics about test set
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_test,
                    SUM(CASE WHEN in_coach420 IS TRUE THEN 1 ELSE 0 END) as coach420_count,
                    SUM(CASE WHEN in_holo4k IS TRUE THEN 1 ELSE 0 END) as holo4k_count,
                    SUM(CASE WHEN in_coach420 IS TRUE AND in_holo4k IS TRUE THEN 1 ELSE 0 END) as both_count
                FROM processed_datasets 
                WHERE split = 'test'
            """)
            test_stats = cursor.fetchone()
            
            print("\nTest set statistics:")
            print(f"Total test samples: {test_stats[0]}")
            print(f"From COACH420: {test_stats[1]}")
            print(f"From Holo4K: {test_stats[2]}")
            print(f"In both datasets: {test_stats[3]}")
            
            conn.commit()
            cursor.close()
            
            print("\nSuccessfully assigned data to train/val/test sets")
            
    except Exception as e:
        print(f"Error adding split column: {str(e)}")
        raise e

def check_overlap():
    """Add and populate in_coach420 and in_holo4k columns to processed_datasets table"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            
            # Add new columns if they don't exist
            cursor.execute("""
                ALTER TABLE processed_datasets 
                ADD COLUMN IF NOT EXISTS in_coach420 BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS in_holo4k BOOLEAN DEFAULT FALSE
            """)
            
            # Get all unique protein names from processed_datasets
            cursor.execute("SELECT DISTINCT pname FROM processed_datasets")
            processed_proteins = {row[0] for row in cursor.fetchall()}
            
            # Get all protein names from COACH420 (removing chain IDs)
            cursor.execute("SELECT name FROM coach420_proteins")
            coach420_proteins = {row[0][:-1] for row in cursor.fetchall()}
            
            # Get all protein names from Holo4K
            cursor.execute("SELECT name FROM holo4k_proteins")
            holo4k_proteins = {row[0] for row in cursor.fetchall()}
            
            # Update in_coach420 column
            for protein in processed_proteins:
                is_in_coach420 = protein in coach420_proteins
                cursor.execute("""
                    UPDATE processed_datasets 
                    SET in_coach420 = %s 
                    WHERE pname = %s
                """, (is_in_coach420, protein))
            
            # Update in_holo4k column
            for protein in processed_proteins:
                is_in_holo4k = protein in holo4k_proteins
                cursor.execute("""
                    UPDATE processed_datasets 
                    SET in_holo4k = %s 
                    WHERE pname = %s
                """, (is_in_holo4k, protein))
            
            # Print statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN in_coach420 THEN 1 ELSE 0 END) as coach420_count,
                    SUM(CASE WHEN in_holo4k THEN 1 ELSE 0 END) as holo4k_count,
                    SUM(CASE WHEN in_coach420 AND in_holo4k THEN 1 ELSE 0 END) as both_count
                FROM processed_datasets
            """)
            stats = cursor.fetchone()
            
            print("\nDataset Origin Statistics:")
            print(f"Total proteins: {stats[0]}")
            print(f"Proteins in COACH420: {stats[1]}")
            print(f"Proteins in Holo4K: {stats[2]}")
            print(f"Proteins in both datasets: {stats[3]}")
            
            conn.commit()
            cursor.close()
            
            print("\nSuccessfully added and populated dataset origin columns")
            
    except Exception as e:
        print(f"Error adding dataset origin columns: {str(e)}")
        raise e

if __name__ == "__main__":
    # Use multiprocessing Pool to run the parallel processing
    parallel_preprocess(num_workers=12, batch_size=10)
    check_overlap()
    set_split()
    test_pocket_dataset()
#%%
