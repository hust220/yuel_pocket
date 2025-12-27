import pandas as pd
import os

def main():
    # Paths
    pdbbind_lst_path = '/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/pdbbind/INDEX_general_PL.2020R1.lst'
    # Using the absolute path found for the PLINDER split file
    plinder_split_path = '/home/tyq4zn/scratch/codes/yuel_pocket/data/plinder/data/2024-06/v2/splits/split.parquet'

    # 1. Read PDBBind IDs
    print(f"Reading PDBBind IDs from {pdbbind_lst_path}...")
    pdbbind_ids = set()
    with open(pdbbind_lst_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) > 0:
                # Assuming the PDB ID is the first column based on standard PDBBind format
                # Will verify after reading the file header in the next step, but typically it is column 0
                pdb_id = parts[0]
                if len(pdb_id) == 4:
                    pdbbind_ids.add(pdb_id.lower())
    
    print(f"Found {len(pdbbind_ids)} PDB IDs in PDBBind.")

    # 2. Read PLINDER Train IDs
    print(f"Reading PLINDER split from {plinder_split_path}...")
    try:
        df = pd.read_parquet(plinder_split_path)
        # Filter for training set
        train_df = df[df['split'] == 'train']
        
        # Extract PDB IDs from 'system_id' (assuming format PDBID_...)
        # We take the first 4 characters of the system_id
        plinder_train_ids = set(train_df['system_id'].apply(lambda x: x[:4].lower()))
        
        print(f"Found {len(plinder_train_ids)} unique PDB IDs in PLINDER training set.")
        
    except Exception as e:
        print(f"Error reading PLINDER parquet file: {e}")
        return

    # 3. Find IDs in PDBBind but NOT in PLINDER Train
    missing_in_train = pdbbind_ids - plinder_train_ids
    sorted_missing_ids = sorted(list(missing_in_train))
    
    print(f"Found {len(missing_in_train)} PDB IDs present in PDBBind but NOT in PLINDER Train.")
    
    # 4. Check these against PLINDER Test
    test_df = df[df['split'] == 'test']
    plinder_test_ids = set(test_df['system_id'].apply(lambda x: x[:4].lower()))
    
    val_df = df[df['split'] == 'val']
    plinder_val_ids = set(val_df['system_id'].apply(lambda x: x[:4].lower()))
    
    common_in_test = missing_in_train.intersection(plinder_test_ids)
    common_in_val = missing_in_train.intersection(plinder_val_ids)
    
    print(f"Out of those {len(missing_in_train)} IDs:")
    print(f"  - {len(common_in_test)} are in PLINDER Test set.")
    print(f"  - {len(common_in_val)} are in PLINDER Validation set.")
    print(f"  - {len(missing_in_train) - len(common_in_test) - len(common_in_val)} are not in PLINDER at all (or in other splits).")

    # 5. Save to files
    # Save missing from train
    output_missing = 'pdbbind_not_in_plinder_train.txt'
    with open(output_missing, 'w') as f:
        for pid in sorted_missing_ids:
            f.write(f"{pid}\n")
    print(f"Saved IDs missing from train to {output_missing}")
    
    # Save those found in test
    if common_in_test:
        output_test = 'pdbbind_in_plinder_test.txt'
        with open(output_test, 'w') as f:
            for pid in sorted(list(common_in_test)):
                f.write(f"{pid}\n")
        print(f"Saved IDs found in PLINDER Test to {output_test}")
        
    # Save those found in val
    if common_in_val:
        output_val = 'pdbbind_in_plinder_val.txt'
        with open(output_val, 'w') as f:
            for pid in sorted(list(common_in_val)):
                f.write(f"{pid}\n")
        print(f"Saved IDs found in PLINDER Validation to {output_val}")

if __name__ == '__main__':
    main()
