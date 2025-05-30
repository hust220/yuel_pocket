import argparse
import pickle
from multiprocessing import Pool
from functools import partial
from rdkit import Chem
from tqdm import tqdm
import psycopg2
from io import StringIO, BytesIO
import sys
import numpy as np
from Bio.PDB import PDBParser
sys.path.append('../../')
from src import const
from src.db_utils import db_connection # Import the centralized db_connection
import os
import json

# Disable RDKit warnings
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

def get_protein(protein_name):
    protein = None
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pdb FROM proteins WHERE name = %s", (protein_name,))
            protein_data = cursor.fetchone()
            cursor.close()
            if protein_data and protein_data[0] is not None:
                protein = StringIO(protein_data[0].tobytes().decode('utf-8'))
    except psycopg2.Error as e:
        print(f"Database error in get_protein for {protein_name}: {str(e)}")
    except Exception as e:
        print(f"Error getting protein {protein_name}: {str(e)}")
    return protein

def aa_one_hot(residue):
    n = const.N_RESIDUE_TYPES
    one_hot = np.zeros(n)
    if residue not in const.RESIDUE2IDX:
        residue = 'UNK'
    one_hot[const.RESIDUE2IDX[residue]] = 1
    return one_hot

def parse_complex(mol, pdb_path, pocket_threshold=6):
    struct = PDBParser(QUIET=True).get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []
    protein_pos = []
    protein_one_hot = []
    is_pocket = []
    protein_backbone = []

    ir1, ir0 = 0, -1
    last_pos = None
    for chain in struct.get_chains():
        for residue in chain.get_residues():
            residue_name = residue.get_resname()
            atom_names = [atom.get_name() for atom in residue.get_atoms()]
            if 'CA' in atom_names:
                ca = next(atom for atom in residue.get_atoms() if atom.get_name() == 'CA')
                atom_coords.extend([atom.get_coord() for atom in residue.get_atoms()])
                residue_ids.extend([ir1] * len(atom_names))

                pos = ca.get_coord().tolist()
                protein_pos.append(pos)
                protein_one_hot.append(aa_one_hot(residue_name))

                if last_pos is not None:
                    if np.linalg.norm(np.array(last_pos) - np.array(pos)) < 4.1:
                        protein_backbone.append([ir0, ir1])
                last_pos = pos
                ir0 += 1
                ir1 += 1

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    mol_atom_coords = mol.GetConformer().GetPositions()

    protein_pos = np.array(protein_pos)
    protein_ligand_distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    protein_ligand_residues = np.unique(residue_ids[np.where(protein_ligand_distances.min(axis=1) <= pocket_threshold)[0]])

    is_pocket = [id in protein_ligand_residues for id in range(len(protein_pos))] # Shape: (N,)
    is_pocket = np.array(is_pocket)

    return protein_pos, np.array(protein_one_hot), np.array(protein_backbone), is_pocket

def get_protein_contacts(protein_name):
    contacts = []
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT contacts FROM proteins WHERE name = %s", (protein_name,))
            result = cursor.fetchone()
            cursor.close()
            if result and result[0] is not None:
                protein_contacts_data = result[0]
                contents = protein_contacts_data.tobytes().decode('utf-8')
                lines = contents.split('\n')
                for line in lines:
                    if line:
                        parts = line.split()
                        contacts.append([int(parts[0]), int(parts[1]), float(parts[2])])
    except psycopg2.Error as e:
        print(f"Database error getting protein contacts for {protein_name}: {str(e)}")
    except Exception as e:
        print(f"Error processing protein contacts for {protein_name}: {str(e)}")
    return np.array(contacts)

def atom_one_hot(atom):
    n = const.N_ATOM_TYPES
    one_hot = np.zeros(n)
    if atom not in const.ATOM2IDX:
        atom = 'X'
    one_hot[const.ATOM2IDX[atom]] = 1
    return one_hot

def bond_one_hot(bond):
    one_hot = [0 for i in range(const.N_RDBOND_TYPES)]
    
    # Set the appropriate index to 1
    bond_type = bond.GetBondType()
    if bond_type in const.RDBOND2IDX:
        one_hot[const.RDBOND2IDX[bond_type]] = 1
    else:
        raise Exception('Unknown bond type {}'.format(bond_type))
        
    return one_hot

def parse_molecule(mol):
    atom_one_hots = []
    for atom in mol.GetAtoms():
        atom_one_hots.append(atom_one_hot(atom.GetSymbol()))

    # if mol has no conformer, positions is 0
    if mol.GetNumConformers() == 0:
        positions = np.zeros((mol.GetNumAtoms(), 3))
    else:
        positions = mol.GetConformer().GetPositions()

    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        one_hot = bond_one_hot(bond)
        bonds.append([i, j] + one_hot)
        bonds.append([j, i] + one_hot)

    return positions, np.array(atom_one_hots), np.array(bonds)

def get_ligands_batch(ligand_ids):
    """Batch fetch ligands from database"""
    ligands = {}
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            # Convert list to tuple for IN clause
            cursor.execute("""
                SELECT id, name, protein_name, mol 
                FROM ligands 
                WHERE id = ANY(%s)
            """, (ligand_ids,))
            for row in cursor.fetchall():
                ligand_id, name, protein_name, mol_bytes = row
                if mol_bytes is not None:
                    ligands[ligand_id] = (name, protein_name, mol_bytes.tobytes().decode('utf-8'))
            cursor.close()
    except psycopg2.Error as e:
        print(f"Database error in get_ligands_batch: {str(e)}")
    except Exception as e:
        print(f"Error processing ligands batch: {str(e)}")
    return ligands

def process_ligands(ligand_ids):
    """Process a list of ligand IDs assigned to this worker"""
    processed_data = []
    
    # Batch fetch all ligands first
    ligands = get_ligands_batch(ligand_ids)
    
    for ligand_id in ligand_ids:
        try:
            if ligand_id not in ligands:
                continue
                
            ligand_name, protein_name, mol_text = ligands[ligand_id]
            
            mol = Chem.MolFromMolBlock(mol_text, sanitize=False, removeHs=False, strictParsing=False)
            
            if not mol:
                continue

            mol.SetProp('_Name', ligand_name)
            mol_smi = Chem.MolToSmiles(mol)
            mol_pos, mol_one_hot, mol_bonds = parse_molecule(mol)
        
            protein_file = get_protein(protein_name)
            if not protein_file:
                continue
            protein_contacts = get_protein_contacts(protein_name)
            protein_pos, protein_one_hot, protein_backbone, is_pocket = parse_complex(mol, protein_file)

            protein_size = len(protein_pos)
            molecule_size = len(mol_pos)
            if protein_size == 0 or molecule_size == 0 or protein_size >= 1000 or len(protein_contacts) == 0:
                continue

            processed_data.append((
                ligand_name,
                protein_name,
                pickle.dumps(mol_pos),
                pickle.dumps(mol_one_hot),
                pickle.dumps(mol_bonds),
                pickle.dumps(protein_pos),
                pickle.dumps(protein_one_hot),
                pickle.dumps(protein_contacts),
                pickle.dumps(protein_backbone),
                pickle.dumps(is_pocket),
                mol_smi,
                protein_size,
                molecule_size
            ))
        except Exception as e:
            print(f"Error processing ligand {ligand_id}: {str(e)}")
            continue
    
    # Upload all processed data to database
    if processed_data:
        try:
            with db_connection() as conn:
                cursor = conn.cursor()                
                # Then insert new entries
                cursor.executemany("""
                    INSERT INTO raw_datasets (
                        ligand_name, protein_name, molecule_pos, molecule_one_hot,
                        molecule_bonds, protein_pos, protein_one_hot, protein_contacts,
                        protein_backbone, is_pocket, smiles, protein_size, molecule_size
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ligand_name, protein_name) DO NOTHING
                """, processed_data)
                conn.commit()
                cursor.close()
                return len(processed_data)
        except psycopg2.Error as e:
            print(f"Database error inserting datasets: {str(e)}")
        except Exception as e:
            print(f"Error inserting datasets: {str(e)}")
    
    return 0

def get_all_ligand_ids():
    ids = []
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM ligands ORDER BY id")
            ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
    except psycopg2.Error as e:
        print(f"Database error in get_all_ligand_ids: {str(e)}")
    except Exception as e:
        print(f"Error getting all ligand IDs: {str(e)}")
    return ids

def get_ligand(ligand_id):
    ligand = None
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, protein_name, mol FROM ligands WHERE id = %s",
                (ligand_id,)
            )
            ligand_data = cursor.fetchone()
            cursor.close()
            if ligand_data:
                name, protein_name, mol_bytes = ligand_data
                if mol_bytes is not None:
                    ligand = (name, protein_name, mol_bytes.tobytes().decode('utf-8'))
    except psycopg2.Error as e:
        print(f"Database error in get_ligand for ID {ligand_id}: {str(e)}")
    except Exception as e:
        print(f"Error getting ligand {ligand_id}: {str(e)}")
    return ligand

def create_raw_datasets_table():
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_datasets (
                    id SERIAL PRIMARY KEY,
                    ligand_name TEXT,
                    protein_name TEXT,
                    molecule_pos BYTEA,
                    molecule_one_hot BYTEA,
                    molecule_bonds BYTEA,
                    protein_pos BYTEA,
                    protein_one_hot BYTEA,
                    protein_contacts BYTEA,
                    protein_backbone BYTEA,
                    is_pocket BYTEA,
                    smiles TEXT,
                    protein_size INTEGER,
                    molecule_size INTEGER,
                    UNIQUE(ligand_name, protein_name)
                )
            """)
            conn.commit()
            cursor.close()
    except psycopg2.Error as e:
        print(f"Database error creating raw_datasets table: {str(e)}")
        raise e

def main(args):
    # Create the raw_datasets table if it doesn't exist
    create_raw_datasets_table()
    
    all_ids = get_all_ligand_ids()
    if args.max_molecules:
        all_ids = all_ids[:args.max_molecules]
    
    print(f"Total {len(all_ids)} molecules to process")

    # Pre-assign ligand IDs to each worker
    chunk_size = 20
    chunks = [all_ids[i:i + chunk_size] for i in range(0, len(all_ids), chunk_size)]
    
    processed_count = 0
    
    with Pool(args.num_workers) as pool:
        with tqdm(total=len(all_ids), desc="Processing molecules") as pbar:
            for result in pool.imap_unordered(process_ligands, chunks):
                processed_count += result
                pbar.update(chunk_size)
                pbar.set_postfix({
                    "success": f"{processed_count}/{pbar.n}",
                })

    print(f"Successfully processed and stored {processed_count} molecules in the database")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=24, help='Number of worker processes')
    parser.add_argument('--max_molecules', type=int, default=None, help='Maximum number of molecules to process')
    args = parser.parse_args()

    main(args)
