import os
import shutil
import glob
from pathlib import Path

def main():
    # Configuration
    # Adjust path to absolute based on user context
    workspace_root = Path('/home/tyq4zn/scratch/codes/yuel_pocket')
    id_list_path = workspace_root / 'benchmark/pdbbind/pdbbind_not_in_plinder_train.txt'
    dest_dir = workspace_root / 'benchmark/pdbbind/test5902'
    source_root = Path('/home/tyq4zn/scratch/datasets/pdbbind/P-L')

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination directory created: {dest_dir}")

    # Read ID list
    if not id_list_path.exists():
        print(f"Error: ID list file found at {id_list_path}")
        return

    with open(id_list_path, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(pdb_ids)} PDB IDs to copy.")

    # Get year directories
    year_dirs = [d for d in source_root.iterdir() if d.is_dir()]
    print(f"Found {len(year_dirs)} year directories in {source_root}: {[d.name for d in year_dirs]}")

    # Copy files
    success_count = 0
    not_found_count = 0
    
    for i, pdb_id in enumerate(pdb_ids):
        found = False
        for year_dir in year_dirs:
            # Check if likely path exists
            # Case insensitive check: PDB IDs are typically lowercase in folders but let's be safe
            # Based on 'ls' output, folders are lowercase
            pdb_folder = year_dir / pdb_id
            
            if pdb_folder.exists():
                found = True
                
                # Source files
                protein_src = pdb_folder / f"{pdb_id}_protein.pdb"
                ligand_src = pdb_folder / f"{pdb_id}_ligand.sdf"
                
                # Destination files
                # Flatten structure: just copy files to test5902/
                # Or keep ID folder structure?
                # User said "copy ... to test5902 folder". Usually implies flat list or standard ID naming.
                # Given standard benchmark scripts usually look for {id}_protein.pdb in the folder, 
                # I will copy files directly to test5902/
                
                protein_dst = dest_dir / f"{pdb_id}_protein.pdb"
                ligand_dst = dest_dir / f"{pdb_id}_ligand.sdf"
                
                # Copy Protein
                if protein_src.exists():
                    shutil.copy2(protein_src, protein_dst)
                else:
                    print(f"Warning: Protein file missing for {pdb_id} at {protein_src}")
                
                # Copy Ligand
                if ligand_src.exists():
                    shutil.copy2(ligand_src, ligand_dst)
                else:
                    print(f"Warning: Ligand file missing for {pdb_id} at {ligand_src}")
                
                success_count += 1
                break
        
        if not found:
            not_found_count += 1
            # Optional: print only if strict debugging needed
            # print(f"Could not find folder for {pdb_id}")
            pass

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(pdb_ids)} IDs...")

    print("-" * 30)
    print(f"Copying complete.")
    print(f"Successfully located folders: {success_count}")
    print(f"Not found: {not_found_count}")

if __name__ == '__main__':
    main()
