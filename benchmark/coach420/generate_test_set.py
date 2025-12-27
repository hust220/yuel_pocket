import os
import sys
from pathlib import Path
from zipfile import ZipFile
import io

# Add project root to sys.path
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT))

from src.pdb_utils import Structure

# List of common ions and solvents to exclude from being considered as ligands
BLACKLIST = {
    'HOH', 'WAT', 'TIP3', 'H2O', 'SOL',  # Water
    'MG', 'MN', 'CA', 'ZN', 'CL', 'NA', 'K', 'FE', 'CU', 'CO', # Ions
    'SO4', 'PO4', 'NO3', 'CO3', # Polyatomic ions
    'EDO', 'GOL', 'DMS', 'PEG', 'ACT', 'FOR' # Solvents / additives
}

def generate_coach420_set():
    current_dir = Path(__file__).parent
    output_dir = current_dir / "all"
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = current_dir / "coach420.zip"
    if not zip_path.exists():
        print(f"Error: Zip file not found at {zip_path}")
        return

    print(f"Extracting and processing COACH420 from {zip_path}...")
    
    with ZipFile(zip_path, 'r') as zf:
        # Find PDB files in the zip
        pdb_files = [name for name in zf.namelist() if name.lower().endswith('.pdb')]
        print(f"Found {len(pdb_files)} PDB files in zip.")

        processed_count = 0
        pair_count = 0

        for pdb_zip_path in pdb_files:
            pdb_id = Path(pdb_zip_path).stem
            
            with zf.open(pdb_zip_path) as f:
                pdb_text = f.read().decode("utf-8", "ignore")
            
            try:
                struct = Structure()
                struct.read(io.StringIO(pdb_text), skip_water=True)
            except Exception as e:
                print(f"Error parsing {pdb_id}: {e}")
                continue

            protein_atoms = []
            # Keep track of ligands by (res_name, chain_id, res_id) to handle multiple instances
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
                print(f"Warning: No protein atoms found in {pdb_id}")
                continue

            # Save Protein
            protein_path = output_dir / f"{pdb_id}_protein.pdb"
            with open(protein_path, 'w') as f:
                for atom in protein_atoms:
                    f.write(atom.to_pdb())
            
            # Save each unique ligand
            unique_ligands_by_name = {}
            for (res_name, ch, rid), atoms in ligands.items():
                if res_name not in unique_ligands_by_name:
                    unique_ligands_by_name[res_name] = []
                unique_ligands_by_name[res_name].append(atoms)

            for res_name, atom_list_list in unique_ligands_by_name.items():
                # If multiple instances of same ligand name, we can save them separately or merged
                # Usually COACH420 evaluates per ligand instance or per ligand name.
                # Let's save them with instance suffix if multiple
                for i, atoms in enumerate(atom_list_list):
                    suffix = f"_{i+1}" if len(atom_list_list) > 1 else ""
                    lig_path = output_dir / f"{pdb_id}_{res_name}{suffix}_ligand.pdb"
                    with open(lig_path, 'w') as f:
                        for atom in atoms:
                            f.write(atom.to_pdb())
                    pair_count += 1
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} PDBs...")

    print(f"\nFinished. Successfully processed {processed_count} PDBs.")
    print(f"Total protein-ligand pairs generated: {pair_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    generate_coach420_set()
