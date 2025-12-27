import os
import shutil
import random
from pathlib import Path

def main():
    # Configuration
    seed = 42
    random.seed(seed)
    
    workspace_root = Path('/home/tyq4zn/scratch/codes/yuel_pocket')
    source_dir = workspace_root / 'benchmark/pdbbind/test5902'
    dest_dir = workspace_root / 'benchmark/pdbbind/test500'
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination directory created: {dest_dir}")
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    # Identify all PDB IDs in source
    # We look for *_protein.pdb files to identify unique systems
    protein_files = list(source_dir.glob('*_protein.pdb'))
    pdb_ids = [f.name.replace('_protein.pdb', '') for f in protein_files]
    
    total_systems = len(pdb_ids)
    print(f"Found {total_systems} systems in {source_dir}")
    
    if total_systems < 500:
        print(f"Warning: Source has fewer than 500 systems ({total_systems}). Copying all.")
        selected_ids = pdb_ids
    else:
        selected_ids = random.sample(pdb_ids, 500)
        print(f"Randomly selected 500 systems.")

    # Copy files
    success_count = 0
    
    for pdb_id in selected_ids:
        # Define source files
        p_src = source_dir / f"{pdb_id}_protein.pdb"
        l_src = source_dir / f"{pdb_id}_ligand.sdf"
        
        # Define dest files
        p_dst = dest_dir / f"{pdb_id}_protein.pdb"
        l_dst = dest_dir / f"{pdb_id}_ligand.sdf"
        
        # Copy
        if p_src.exists() and l_src.exists():
            shutil.copy2(p_src, p_dst)
            shutil.copy2(l_src, l_dst)
            success_count += 1
        else:
            print(f"Warning: Missing files for {pdb_id}")
            
    print("-" * 30)
    print(f"Created test500 with {success_count} systems.")

if __name__ == '__main__':
    main()
