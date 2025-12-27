import os
import sys
import argparse
from pathlib import Path
from zipfile import ZipFile
import io

# Add project root to sys.path
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT))

from src.pdb_utils import Structure, Atom

def parse_ds_file(ds_content):
    """
    Parse the .ds file content.
    Format: path/pdbid.pdb  CODE1,CODE2,...
    Returns a dict mapping pdb_id (lower) to a list of ligand codes.
    """
    ligand_map = {}
    for line in ds_content.splitlines():
        line = line.strip()
        if not line or line.startswith('HEADER:'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            pdb_path = parts[0]
            # Extract pdb_id from path like "holo4k/121p.pdb"
            pdb_id = Path(pdb_path).stem.lower()
            codes = parts[1].split(',')
            ligand_map[pdb_id] = [c.strip() for c in codes]
    return ligand_map

def get_target_pdbs(details_path):
    """
    Read target PDB IDs from the overlap details file.
    """
    target_pdbs = set()
    if not os.path.exists(details_path):
        print(f"Error: Details file not found at {details_path}")
        return target_pdbs

    with open(details_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith('==='):
            continue
        # Items are comma separated
        parts = line.split(',')
        for p in parts:
            p = p.strip().lower()
            if p:
                target_pdbs.add(p)
    return target_pdbs

def generate_test340():
    current_dir = Path(__file__).parent
    output_dir = current_dir / "test340"
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = current_dir / "holo4k.zip"
    details_path = current_dir / "holo4k_plinder_overlap_details.txt"

    if not zip_path.exists():
        print(f"Error: Zip file not found at {zip_path}")
        return

    target_pdbs = get_target_pdbs(details_path)
    if not target_pdbs:
        print("No target PDBs found.")
        return
    print(f"Found {len(target_pdbs)} target PDBs.")

    with ZipFile(zip_path, 'r') as zf:
        # 1. Find the .ds file in the zip
        ds_file_name = None
        for name in zf.namelist():
            if name.endswith('.ds'):
                ds_file_name = name
                break
        
        if not ds_file_name:
            print("Error: .ds file not found in zip.")
            return

        print(f"Reading {ds_file_name}...")
        with zf.open(ds_file_name) as f:
            ds_content = f.read().decode("utf-8", "ignore")
        
        ligand_map = parse_ds_file(ds_content)

        # 2. Extract and process each PDB
        success_count = 0
        pair_count = 0

        for pdb_id in target_pdbs:
            # Construct path in zip
            # Note: zip paths were home/tyq4zn/scratch/datasets/holo4k/pdbid.pdb
            pdb_zip_path = f"home/tyq4zn/scratch/datasets/holo4k/{pdb_id}.pdb"
            
            if pdb_zip_path not in zf.namelist():
                # Try fallback if naming varies (some might be uppercase in zip?)
                pdb_zip_path = None
                for name in zf.namelist():
                    if name.lower().endswith(f"/{pdb_id}.pdb"):
                        pdb_zip_path = name
                        break
            
            if not pdb_zip_path:
                print(f"Warning: {pdb_id}.pdb not found in zip.")
                continue

            with zf.open(pdb_zip_path) as f:
                pdb_text = f.read().decode("utf-8", "ignore")
            
            # Use pdb_utils
            try:
                struct = Structure()
                struct.read(io.StringIO(pdb_text), skip_water=True)
            except Exception as e:
                print(f"Error parsing {pdb_id}: {e}")
                continue

            lig_codes = ligand_map.get(pdb_id, [])
            if not lig_codes:
                print(f"Warning: No ligand codes for {pdb_id} in .ds file.")
                continue

            # Separate Protein and Ligands
            # Protein: all ATOM records (residues where is_hetatm is False)
            # Ligand: residues with matching name
            
            protein_atoms = []
            ligand_atoms_dict = {code: [] for code in lig_codes}
            
            for model in struct:
                for chain in model:
                    for residue in chain:
                        if not residue.is_hetatm():
                            protein_atoms.extend(residue.atoms)
                        elif residue.res_name in ligand_atoms_dict:
                            ligand_atoms_dict[residue.res_name].extend(residue.atoms)

            if not protein_atoms:
                print(f"Warning: No protein atoms found in {pdb_id}")
                continue

            # Save Protein
            protein_path = output_dir / f"{pdb_id}_protein.pdb"
            with open(protein_path, 'w') as f:
                for atom in protein_atoms:
                    f.write(atom.to_pdb())
            
            # Save each ligand
            for code, atoms in ligand_atoms_dict.items():
                if not atoms:
                    # Some ligands might be missing in the actual PDB despite being in .ds
                    continue
                
                lig_path = output_dir / f"{pdb_id}_{code}_ligand.pdb"
                with open(lig_path, 'w') as f:
                    for atom in atoms:
                        f.write(atom.to_pdb())
                pair_count += 1
            
            success_count += 1
            # print(f"Processed {pdb_id}")

    print(f"Finished. Successfully processed {success_count} PDBs.")
    print(f"Total protein-ligand pairs generated: {pair_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    generate_test340()
