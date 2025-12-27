import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile
from io import StringIO, BytesIO
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, '../../'))
if proj_root not in sys.path:
    sys.path.append(proj_root)

from src.pos_sc.model import YuelPocket
from src.pos_sc.dataset import PocketDataset, build_graph, parse_protein, parse_molecule, SYSTEMS_DIR, SPLIT_PATH
from src.lightning import LightningWrapper
from src.utils import pick_latest
from src import const

def main():
    parser = argparse.ArgumentParser(description="Find hard decoys using YuelPocket pos_sc model.")
    parser.add_argument('--output', type=str, default=os.path.join(current_dir, 'hard_decoys.zip'), help="Output zip file")
    parser.add_argument('--model', type=str, default=None, help="Path to model checkpoint")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num_chunks', type=int, default=1, help="Total number of chunks to split data into")
    parser.add_argument('--chunk', type=int, default=0, help="Current chunk index (0 to num_chunks-1)")
    args = parser.parse_args()

    # 1. Load Model
    if args.model:
        model_path = args.model
    else:
        # Try to find latest pos_sc checkpoint
        target = "models/plinder_pos_sc_bs*"
        patterns = [
            os.path.join(proj_root, target, "**/*.ckpt"),
            os.path.join(proj_root, target)
        ]
        model_path = pick_latest(patterns)

    if not model_path:
        print("Error: Could not find pos_sc model checkpoint.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    try:
        model = LightningWrapper.load_from_checkpoint(
            model_path,
            map_location=args.device,
            model_class=YuelPocket,
            weights_only=False,
            strict=False
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    model.to(args.device)
    model.eval()

    # 2. Load Systems
    if not os.path.exists(SPLIT_PATH):
        print(f"Error: Split file not found at {SPLIT_PATH}")
        sys.exit(1)

    print(f"Loading split from {SPLIT_PATH}...")
    table = pq.read_table(SPLIT_PATH, columns=['system_id'])
    system_ids = table.to_pandas()['system_id'].unique().tolist()
    if args.limit:
        system_ids = system_ids[:args.limit]
    
    # Chunking logic
    if args.num_chunks > 1:
        chunk_size = (len(system_ids) + args.num_chunks - 1) // args.num_chunks
        start = args.chunk * chunk_size
        end = min(start + chunk_size, len(system_ids))
        system_ids = system_ids[start:end]
        print(f"Processing chunk {args.chunk}/{args.num_chunks} (IDs {start} to {end})")
    
    print(f"Processing {len(system_ids)} systems.")

    # 3. Process
    # PocketDataset.__init__ loads random ligands and setup paths
    dataset = PocketDataset(split='train')
    
    with ZipFile(args.output, 'w') as zf_out:
        pbar = tqdm(system_ids)
        for system_id in pbar:
            try:
                # Load data from PLINDER buckets
                res = dataset._read_from_zip(system_id)
                if res is None: continue
                raw_id, receptor_pdb, ligand_mol = res
                
                if ligand_mol is None: continue
                
                mol_pos, mol_one_hot, mol_bonds = parse_molecule(ligand_mol)
                prot_pos, prot_one_hot, prot_contacts, _ = parse_protein(receptor_pdb)
                
                # Try to load existing SAS points (if precomputed)
                sas_points = None
                if dataset.sas_zip is not None:
                    try:
                        with dataset.sas_zip.open(f"{system_id}.npy") as f:
                            sas_points = np.load(f).astype(np.float32)
                    except: pass
                
                if sas_points is None or len(sas_points) == 0:
                    continue

                # Define is_pocket and is_decoy for ALL SAS points based on distance to ligand
                ligand_centroid = np.mean(mol_pos, axis=0, keepdims=True)
                dists = cdist(sas_points, ligand_centroid).flatten()
                
                is_pocket_mask = dists < 4.0
                is_decoy_mask = dists >= 4.0
                
                if not np.any(is_pocket_mask) or not np.any(is_decoy_mask):
                    continue

                # Build Graph for Inference (pick_samples=False to use all SAS points as probes)
                g = build_graph(
                    protein_name=system_id,
                    ligand_name=system_id,
                    mol_pos=mol_pos,
                    mol_h=mol_one_hot,
                    mol_bonds=mol_bonds,
                    prot_pos=prot_pos,
                    prot_h=prot_one_hot,
                    prot_contacts=prot_contacts,
                    sas_points=sas_points,
                    pick_samples=False 
                )
                g.to(args.device)

                # Inference
                with torch.no_grad():
                    logits = model.sample_chain(g)
                
                # Extract predicted scores for probes (last N_SAS nodes in the graph)
                n_probes = len(sas_points)
                probe_scores = logits[-n_probes:].cpu().numpy()

                pocket_scores = probe_scores[is_pocket_mask]
                min_pocket_score = np.min(pocket_scores)
                
                hard_mask = is_decoy_mask & (probe_scores > min_pocket_score)
                hard_indices = np.where(hard_mask)[0]
                pocket_indices = np.where(is_pocket_mask)[0]
                
                # Combine pocket points and hard decoys
                selected_indices = np.concatenate([pocket_indices, hard_indices])
                if len(selected_indices) > 0:
                    selected_points = sas_points[selected_indices]
                    # Save both pocket points and hard decoys to the zip archive
                    with BytesIO() as bio:
                        np.save(bio, selected_points)
                        bio.seek(0)
                        zf_out.writestr(f"{system_id}.npy", bio.read())
                
            except Exception as e:
                # print(f"Error processing {system_id}: {e}")
                continue

    print(f"Finished. Saved hard decoys to {args.output}")

if __name__ == "__main__":
    main()
