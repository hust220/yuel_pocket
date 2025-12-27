import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def check_overlap():
    # 1. Get COACH420 PDB IDs
    # Filenames are like '148lE.pdb' where first 4 chars is PDB ID
    coach420_dir = Path("~/scratch/datasets/coach420").expanduser()
    if not coach420_dir.exists():
        print(f"Error: COACH420 directory not found at {coach420_dir}")
        return
    
    coach420_pdbs = set()
    for f in coach420_dir.glob("*.pdb"):
        pdb_id = f.stem[:4].lower()
        coach420_pdbs.add(pdb_id)
    
    print(f"Total Unique PDB IDs in COACH420: {len(coach420_pdbs)}")

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
    plinder_removed_pdbs = set(df[df['split'] == 'removed']['pdb_id'])
    all_plinder_pdbs = set(df['pdb_id'])

    # 3. Compare
    in_train = coach420_pdbs.intersection(plinder_train_pdbs)
    in_test = coach420_pdbs.intersection(plinder_test_pdbs)
    in_val = coach420_pdbs.intersection(plinder_val_pdbs)
    in_removed = coach420_pdbs.intersection(plinder_removed_pdbs)
    not_in_plinder = coach420_pdbs - all_plinder_pdbs

    print("\nComparison Results (COACH420 vs PLINDER):")
    print(f"{'Category':<25} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    total = len(coach420_pdbs)
    print(f"{'In PLINDER Train':<25} {len(in_train):<10} {len(in_train)/total:.2%}")
    print(f"{'In PLINDER Test':<25} {len(in_test):<10} {len(in_test)/total:.2%}")
    print(f"{'In PLINDER Val':<25} {len(in_val):<10} {len(in_val)/total:.2%}")
    print(f"{'In PLINDER Removed':<25} {len(in_removed):<10} {len(in_removed)/total:.2%}")
    print(f"{'Not in PLINDER':<25} {len(not_in_plinder):<10} {len(not_in_plinder)/total:.2%}")
    
    check_sum = len(in_train) + len(in_test) + len(in_val) + len(in_removed) + len(not_in_plinder)
    print("-" * 50)
    print(f"{'Total':<25} {check_sum:<10} {check_sum/total:.2%}")

    # 4. Save details
    with open("coach420_plinder_overlap_details.txt", "w") as f:
        f.write("=== COACH420 PDBs in PLINDER TEST ===\n")
        f.write(", ".join(sorted(list(in_test))) + "\n\n")
        f.write("=== COACH420 PDBs NOT IN PLINDER ===\n")
        f.write(", ".join(sorted(list(not_in_plinder))) + "\n")

    print(f"\nDetails saved to {os.getcwd()}/coach420_plinder_overlap_details.txt")

if __name__ == "__main__":
    check_overlap()
