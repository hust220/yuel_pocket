import os
import sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from zipfile import ZipFile
import io

# Add project root to sys.path
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT))

from src.pdb_utils import Structure

BLACKLIST = {
    'HOH', 'WAT', 'TIP3', 'H2O', 'SOL',
    'MG', 'MN', 'CA', 'ZN', 'CL', 'NA', 'K', 'FE', 'CU', 'CO',
    'SO4', 'PO4', 'NO3', 'CO3',
    'EDO', 'GOL', 'DMS', 'PEG', 'ACT', 'FOR'
}

def get_filtered_pdbs():
    # 1. Get COACH420 PDB IDs
    coach420_dir = Path("~/scratch/datasets/coach420").expanduser()
    coach420_pdbs = {f.stem.lower(): f.stem for f in coach420_dir.glob("*.pdb")}
    
    # 2. Get PLINDER Splits
    plinder_split_path = "/sfs/weka/scratch/tyq4zn/codes/yuel_pocket/data/plinder/data/2024-06/v2/splits/split.parquet"
    table = pq.read_table(plinder_split_path, columns=['system_id', 'split'])
    df = table.to_pandas()
    df['pdb_id_lower'] = df['system_id'].apply(lambda x: x.split('__')[0].lower())
    
    plinder_test = set(df[df['split'] == 'test']['pdb_id_lower'])
    plinder_removed = set(df[df['split'] == 'removed']['pdb_id_lower'])
    all_plinder = set(df['pdb_id_lower'])

    target_pdbs_lower = set()
    for stem_lower, original_stem in coach420_pdbs.items():
        pdb_id_4 = stem_lower[:4] # Only first 4 chars for split check
        if pdb_id_4 in plinder_test or pdb_id_4 in plinder_removed or pdb_id_4 not in all_plinder:
            target_pdbs_lower.add(stem_lower)
            
    # Return mapping to original case filenames
    return {coach420_pdbs[l] for l in target_pdbs_lower}

def generate_subset(folder_name):
    current_dir = Path(__file__).parent
    output_dir = current_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = current_dir / "coach420.zip"
    target_stems = get_filtered_pdbs()
    print(f"Targeting {len(target_stems)} unique PDBs (Test/Removed/NotInPlinder).")

    with ZipFile(current_dir / "coach420.zip", 'r') as zf:
        processed_count = 0
        pair_count = 0
        
        for stem in target_stems:
            pdb_zip_path = None
            for name in zf.namelist():
                if Path(name).stem == stem:
                    pdb_zip_path = name
                    break
            
            if not pdb_zip_path:
                continue

            with zf.open(pdb_zip_path) as f:
                pdb_text = f.read().decode("utf-8", "ignore")
            
            try:
                struct = Structure()
                struct.read(io.StringIO(pdb_text), skip_water=True)
            except:
                continue

            protein_atoms = []
            ligands = {}

            for model in struct:
                for chain in model:
                    for residue in chain:
                        if not residue.is_hetatm():
                            protein_atoms.extend(residue.atoms)
                        else:
                            res_name = residue.res_name.strip().upper()
                            if res_name not in BLACKLIST:
                                key = (res_name, residue.chain_id, residue.res_id)
                                ligands[key] = residue.atoms

            if not protein_atoms:
                continue

            # Save Protein
            protein_path = output_dir / f"{stem}_protein.pdb"
            with open(protein_path, 'w') as f:
                for atom in protein_atoms:
                    f.write(atom.to_pdb())
            
            unique_ligands_by_name = {}
            for (res_name, ch, rid), atoms in ligands.items():
                if res_name not in unique_ligands_by_name:
                    unique_ligands_by_name[res_name] = []
                unique_ligands_by_name[res_name].append(atoms)

            for res_name, atom_list_list in unique_ligands_by_name.items():
                for i, atoms in enumerate(atom_list_list):
                    suffix = f"_{i+1}" if len(atom_list_list) > 1 else ""
                    lig_path = output_dir / f"{stem}_{res_name}{suffix}_ligand.pdb"
                    with open(lig_path, 'w') as f:
                        for atom in atoms:
                            f.write(atom.to_pdb())
                    pair_count += 1
            
            processed_count += 1

    print(f"Finished. Generated {pair_count} pairs from {processed_count} proteins in {output_dir}")

if __name__ == "__main__":
    generate_subset("test_removed_notplinder")
