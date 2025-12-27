import os
import glob
from zipfile import ZipFile
from tqdm import tqdm

def merge_zips(output_name, pattern):
    zip_files = sorted(glob.glob(pattern))
    if not zip_files:
        print(f"No files found matching {pattern}")
        return

    print(f"Merging {len(zip_files)} files into {output_name}...")
    
    with ZipFile(output_name, 'w') as out_zf:
        for zf_path in zip_files:
            print(f"Processing {zf_path}...")
            with ZipFile(zf_path, 'r') as in_zf:
                file_list = in_zf.namelist()
                for file_name in tqdm(file_list, desc=os.path.basename(zf_path)):
                    out_zf.writestr(file_name, in_zf.read(file_name))
            
    print(f"Successfully merged into {output_name}")
    
    # Optional: Delete chunks after merging
    # for zf_path in zip_files:
    #     os.remove(zf_path)

if __name__ == "__main__":
    merge_zips("hard_decoys.zip", "hard_decoys_chunk_*.zip")
