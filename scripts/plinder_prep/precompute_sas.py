import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# root is data/plinder/../../ -> yuel_pocket
proj_root = os.path.abspath(os.path.join(current_dir, '../../'))
if proj_root not in sys.path:
    sys.path.append(proj_root)

import argparse
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool
from tqdm import tqdm
from zipfile import ZipFile
from io import StringIO
from threading import Lock

# Import from project
# Ensure src is importable
try:
    from src.positions.dataset import SYSTEMS_DIR, SPLIT_PATH
    from src.pdb_utils import Structure, get_sas_points_shrake_rupley
    from src.positions.config import get_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

from io import BytesIO
import zipfile

def process_system(system_id):
    """
    Worker function to process a single system.
    """
    try:
        # Determine zip path
        bucket = system_id[1:3]
        zip_path = os.path.join(SYSTEMS_DIR, f"{bucket}.zip")
        
        if not os.path.exists(zip_path):
            return None

        # Extract receptor.pdb
        with ZipFile(zip_path, 'r') as zf:
            receptor_path = f"{system_id}/receptor.pdb"
            try:
                with zf.open(receptor_path) as f:
                    receptor_pdb = f.read().decode("utf-8", "ignore")
            except KeyError:
                return None

        # Parse structure
        structure = Structure()
        structure.read(StringIO(receptor_pdb))
        
        if len(structure.models) == 0:
             return None

        # SAS params
        config = get_config()
        probe_radius = config.get('sas_probe_radius', 1.4)
        n_points = config.get('sas_n_points', 15)

        # Calculate SAS
        sas_points, _ = get_sas_points_shrake_rupley(structure, probe_radius=probe_radius, n_points_per_atom=n_points, target_points=None)
        
        if len(sas_points) == 0:
            sas_points = np.zeros((0, 3), dtype=np.float32)
        else:
            sas_points = sas_points.astype(np.float32)
            
        return system_id, sas_points

    except Exception as e:
        # import traceback
        # print(f"Error processing {system_id}: {e}")
        # traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Precompute SAS points for all systems.")
    parser.add_argument('--output', type=str, default=os.path.join(current_dir, 'sas_points.zip'), help="Output zip file")
    parser.add_argument('--workers', type=int, default=32, help="Number of worker processes")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of systems for testing")
    parser.add_argument('--compression', action='store_true', help="Use compression (slower but smaller)")
    args = parser.parse_args()
    
    if not os.path.exists(SPLIT_PATH):
        print(f"Error: Split file not found at {SPLIT_PATH}")
        sys.exit(1)

    print(f"Loading split from {SPLIT_PATH}...")
    table = pq.read_table(SPLIT_PATH, columns=['system_id'])
    system_ids = table.to_pandas()['system_id'].unique().tolist()
    print(f"Found {len(system_ids)} unique systems.")

    if args.limit:
        system_ids = system_ids[:args.limit]
        print(f"Limiting to first {args.limit} systems.")

    # Compression method
    compression = zipfile.ZIP_DEFLATED if args.compression else zipfile.ZIP_STORED
    print(f"Starting processing with {args.workers} workers. Compression: {args.compression}")
    
    processed_count = 0
    
    with ZipFile(args.output, 'w', compression=compression) as zf_out:
        with Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap_unordered(process_system, system_ids, chunksize=10), total=len(system_ids)):
                if res is not None:
                    sys_id, points = res
                    
                    # Save to BytesIO
                    with BytesIO() as bio:
                        np.save(bio, points)
                        bio.seek(0)
                        zf_out.writestr(f"{sys_id}.npy", bio.read())
                    
                    processed_count += 1
                    
    print(f"Successfully processed and saved {processed_count} systems to {args.output}")

if __name__ == "__main__":
    main()
