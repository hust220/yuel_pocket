import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def check_overlap():
    # 1. Get HOLO4K PDB IDs
    holo4k_dir = Path("~/scratch/datasets/holo4k").expanduser()
    if not holo4k_dir.exists():
        print(f"Error: HOLO4K directory not found at {holo4k_dir}")
        return
    
    holo4k_pdbs = set()
    for f in holo4k_dir.glob("*.pdb"):
        holo4k_pdbs.add(f.stem.lower())
    
    print(f"Total Unique PDBs in HOLO4K: {len(holo4k_pdbs)}")

    # 2. Get PLINDER Splits
    plinder_split_path = "/sfs/weka/scratch/tyq4zn/codes/yuel_pocket/data/plinder/data/2024-06/v2/splits/split.parquet"
    if not os.path.exists(plinder_split_path):
        print(f"Error: PLINDER split file not found at {plinder_split_path}")
        return

    print("Reading PLINDER split file...")
    table = pq.read_table(plinder_split_path, columns=['system_id', 'split'])
    df = table.to_pandas()
    
    # Map system_id to pdb_id (first 4 chars)
    df['pdb_id'] = df['system_id'].apply(lambda x: x.split('__')[0].lower())
    
    plinder_train_pdbs = set(df[df['split'] == 'train']['pdb_id'])
    plinder_test_pdbs = set(df[df['split'] == 'test']['pdb_id'])
    plinder_val_pdbs = set(df[df['split'] == 'val']['pdb_id'])
    all_plinder_pdbs = set(df['pdb_id'])

    # 3. Compare
    in_train = holo4k_pdbs.intersection(plinder_train_pdbs)
    in_test = holo4k_pdbs.intersection(plinder_test_pdbs)
    in_val = holo4k_pdbs.intersection(plinder_val_pdbs)
    not_in_plinder = holo4k_pdbs - all_plinder_pdbs

    print("\nComparison Results:")
    print(f"{'Category':<25} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    total = len(holo4k_pdbs)
    print(f"{'In PLINDER Train':<25} {len(in_train):<10} {len(in_train)/total:.2%}")
    print(f"{'In PLINDER Test':<25} {len(in_test):<10} {len(in_test)/total:.2%}")
    print(f"{'In PLINDER Val':<25} {len(in_val):<10} {len(in_val)/total:.2%}")
    print(f"{'Not in PLINDER':<25} {len(not_in_plinder):<10} {len(not_in_plinder)/total:.2%}")

    # 4. Save details
    with open("holo4k_plinder_overlap_details.txt", "w") as f:
        f.write("=== HOLO4K PDBs in PLINDER TEST ===\n")
        f.write(", ".join(sorted(list(in_test))) + "\n\n")
        f.write("=== HOLO4K PDBs NOT IN PLINDER ===\n")
        f.write(", ".join(sorted(list(not_in_plinder))) + "\n")

    print(f"\nDetails saved to {os.getcwd()}/holo4k_plinder_overlap_details.txt")

if __name__ == "__main__":
    check_overlap()
