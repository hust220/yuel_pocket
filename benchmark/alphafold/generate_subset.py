import os
import random
import shutil
import argparse
from pathlib import Path

def generate_subset(source_dir, target_dir, n, seed):
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        print(f"Error: Source directory {source_path} does not exist.")
        return

    # Create target directory if it doesn't exist
    if not target_path.exists():
        os.makedirs(target_path)
        print(f"Created directory: {target_path}")
    else:
        print(f"Directory {target_path} already exists.")

    # Get all files
    files = os.listdir(source_path)

    # Identify unique systems (prefixes)
    # Suffixes are _ligand.sdf and _protein.pdb
    prefixes = set()
    for f in files:
        if f.endswith("_ligand.sdf"):
            prefixes.add(f[:-11]) # remove _ligand.sdf
        elif f.endswith("_protein.pdb"):
            prefixes.add(f[:-12]) # remove _protein.pdb

    sorted_prefixes = sorted(list(prefixes))
    print(f"Found {len(sorted_prefixes)} unique systems in {source_path}.")

    # Randomly select n systems
    if len(sorted_prefixes) < n:
        print(f"Warning: Not enough systems to sample {n}. Using all {len(sorted_prefixes)} systems.")
        selected_prefixes = sorted_prefixes
    else:
        random.seed(seed)
        selected_prefixes = random.sample(sorted_prefixes, n)

    print(f"Selected {len(selected_prefixes)} systems (seed={seed}).")

    # Copy files
    count = 0
    for prefix in selected_prefixes:
        ligand_file = f"{prefix}_ligand.sdf"
        protein_file = f"{prefix}_protein.pdb"
        
        src_lig = source_path / ligand_file
        src_prot = source_path / protein_file
        
        dst_lig = target_path / ligand_file
        dst_prot = target_path / protein_file
        
        # Check both exist before copying to ensure valid pair? Or just copy what exists.
        # Ideally we want pairs.
        if src_lig.exists() and src_prot.exists():
            shutil.copy2(src_lig, dst_lig)
            shutil.copy2(src_prot, dst_prot)
            count += 1
        else:
            if not src_lig.exists():
                print(f"Warning: Missing ligand file for {prefix}")
            if not src_prot.exists():
                print(f"Warning: Missing protein file for {prefix}")
            
    print(f"Successfully copied files for {count} systems to {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random subset of systems.")
    parser.add_argument("--input_dir", type=str, default="test1036_af", help="Input directory containing systems")
    parser.add_argument("--output_dir", type=str, default="test200", help="Output directory for the subset")
    parser.add_argument("--n", type=int, default=200, help="Number of systems to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location if not absolute
    base_dir = Path(__file__).parent
    input_path = base_dir / args.input_dir if not os.path.isabs(args.input_dir) else args.input_dir
    output_path = base_dir / args.output_dir if not os.path.isabs(args.output_dir) else args.output_dir
    
    generate_subset(input_path, output_path, args.n, args.seed)
