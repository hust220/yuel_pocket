import os
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
# Add src to python path implicitly if needed, but since we copy logic we just need imports
# For pdb_line, we can redefine it or import it if PYTHONPATH is set correctly.
# Assuming this script is run where src is importable.
project_root = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.append(str(project_root))

from src.clustering import hill_climbing_cluster

K_NN = 30

def pdb_line(record, atom_id, atom_name, alt_loc, res_name, chain_id, res_id, insertion, x, y, z, occupancy, temp_factor, element, charge):
    return "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format(
        record, atom_id, atom_name, alt_loc, res_name, chain_id, res_id, insertion, x, y, z, occupancy, temp_factor, element, charge
    )

def load_predictions(txt_path):
    points = []
    scores = []
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            start_idx = 0
            if lines and ("Score" in lines[0] or "X" in lines[0]):
                start_idx = 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        x, y, z, s = map(float, parts[:4])
                        points.append([x, y, z])
                        scores.append(s)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
    return np.array(points), np.array(scores)

# cluster_predictions is now imported as hill_climbing_cluster

def save_clusters_pdb(clusters_info, sas_points, probe_scores, pdb_out):
    with open(pdb_out, 'w') as f:
        atom_count = 1
        for cluster in clusters_info:
            cid = cluster['id'] + 1 # 1-based cluster ID
            for idx in cluster['indices']:
                coord = sas_points[idx]
                score = probe_scores[idx]
                line = pdb_line(record="HETATM",
                                atom_id=atom_count,
                                atom_name=' H  ',
                                alt_loc=" ",
                                res_name='PKT',
                                chain_id="A",
                                res_id=cid,
                                insertion=" ",
                                x=coord[0],
                                y=coord[1],
                                z=coord[2],
                                occupancy=1.0,
                                temp_factor=score,
                                element="H",
                                charge="  ")
                f.write(line + "\n")
                atom_count += 1

def process_file(txt_file, k_nn=10):
    txt_path = Path(txt_file)
    if not txt_path.exists():
        print(f"File not found: {txt_path}")
        return

    points, scores = load_predictions(txt_path)
    if len(points) == 0:
        print(f"No valid points found in {txt_path}")
        return

    print(f"Processing {txt_path.name}: {len(points)} points...")
    
    _, clusters_info = hill_climbing_cluster(points, scores, k_nn=k_nn)
    print(f"  Found {len(clusters_info)} clusters.")
    
    if not clusters_info:
        return

    # Keep only top 10 points per cluster by score
    for cluster in clusters_info:
        indices = cluster['indices']
        cluster_scores = scores[indices]
        # Sort indices by scores in descending order
        sorted_indices = indices[np.argsort(cluster_scores)[::-1]]
        cluster['indices'] = sorted_indices[:10]

    # Save Clusters CSV
    cluster_csv = txt_path.with_name(txt_path.stem + "_clusters.csv")
    with open(cluster_csv, 'w') as f:
        f.write("ClusterID,X,Y,Z,Score\n")
        for cluster in clusters_info:
            cid = cluster['id']
            for idx in cluster['indices']:
                pct = points[idx]
                sco = scores[idx] 
                f.write(f"{cid},{pct[0]:.4f},{pct[1]:.4f},{pct[2]:.4f},{sco:.4f}\n")
    
    # Save Clusters PDB
    cluster_pdb = txt_path.with_name(txt_path.stem + "_clusters.pdb")
    save_clusters_pdb(clusters_info, points, scores, cluster_pdb)
    print(f"  Saved clusters to {cluster_pdb.name}")


def main(folder_path, k_nn=10):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    txt_files = list(folder.glob("*_predictions.txt"))
    if not txt_files:
        print("No prediction txt files found.")
        return

    print(f"Found {len(txt_files)} files in {folder}")
    
    for txt_file in txt_files:
        process_file(txt_file, k_nn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster SAS predictions and save as PDB")
    parser.add_argument("folder", help="Folder containing _predictions.txt files")
    parser.add_argument("--k_nn", type=int, default=K_NN, help=f"Number of nearest neighbors for hill climbing (default: {K_NN})")
    args = parser.parse_args()
    
    main(args.folder, args.k_nn)
