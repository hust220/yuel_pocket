import os
import csv
import sys
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt

# Add project root to path to import app logic
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.clustering import hill_climbing_cluster

# Parameter for hill-climbing clustering
K_NN = 30

def calculate_center(atoms):
    """Calculate the geometric center (centroid) of a list of atoms."""
    coords = [atom.get_coord() for atom in atoms]
    if not coords:
        return None
    return np.mean(coords, axis=0)

def calculate_dca(pocket_center, ligand_atoms):
    """
    Calculate Distance to Center of Active site (DCA).
    Distance from pocket center to the closest ligand atom.
    """
    if pocket_center is None or not ligand_atoms:
        return float('inf')
    
    ligand_coords = np.array([atom.get_coord() for atom in ligand_atoms])
    dists = np.linalg.norm(ligand_coords - pocket_center, axis=1)
    return np.min(dists)

def calculate_dcc(pocket_center, ligand_center):
    """
    Calculate Distance to Center of Center (DCC).
    Distance from pocket center to ligand center.
    """
    if pocket_center is None or ligand_center is None:
        return float('inf')
    return np.linalg.norm(pocket_center - ligand_center)

def load_and_cluster_yuelpocket_predictions(txt_path):
    """
    Load YuelPocket predictions from TXT file and run clustering using hill-climbing.
    Returns sorted list of pocket predictions.
    """
    predictions = []
    if not os.path.exists(txt_path):
        return predictions

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
        return []
                
    if not points:
        return []

    points = np.array(points)
    scores = np.array(scores)
    
    # Run Hill-Climbing Clustering
    _, clusters_info = hill_climbing_cluster(points, scores, k_nn=K_NN, min_size=1, max_clusters=100)
    
    rank = 1
    for cluster in clusters_info:
        predictions.append({
            'rank': rank,
            'score': cluster['score'],
            'center': cluster['center']
        })
        rank += 1
    
    return predictions

def plot_success_rates(cutoffs, top1_rates, top3_rates, title, ylabel, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(cutoffs, top1_rates, marker='o', label='Top 1', linewidth=2)
    plt.plot(cutoffs, top3_rates, marker='s', label='Top 3', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Distance Cutoff (Å)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(cutoffs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

def evaluate_predictions(dataset_folder, mode="pos_sc3"):
    base_dir = Path(__file__).parent
    yuel_dir = base_dir / f"{dataset_folder}_yuelpocket_{mode}_predictions"
    dataset_dir = base_dir / dataset_folder
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    ligand_files = list(dataset_dir.glob('*_ligand.pdb'))
    if not ligand_files:
        print("No ligand files found.")
        return

    cutoffs = [4, 5, 6, 7, 8, 9, 10]
    all_dca_top1 = []
    all_dca_top3 = []
    all_dcc_top1 = []
    all_dcc_top3 = []
    
    print(f"{'Protein':<10} {'Ligand':<10} {'Top1_DCA':<10} {'Top1_DCC':<10}")
    print("-" * 55)

    parser = PDBParser(QUIET=True)
    total_cases = 0

    for lig_file in ligand_files:
        parts = lig_file.name.split('_')
        if len(parts) < 3:
            continue
            
        pdb_id = parts[0]
        prot_filename = f"{pdb_id}_protein.pdb"
        pred_txt_name = f"{prot_filename}_predictions.txt"
        pred_txt_path = yuel_dir / pred_txt_name
        
        if not pred_txt_path.exists():
            continue

        try:
            structure = parser.get_structure('ligand', str(lig_file))
            ligand_atoms = list(structure.get_atoms())
        except Exception:
            continue
            
        if not ligand_atoms:
            continue
            
        ligand_center = calculate_center(ligand_atoms)
        
        preds = load_and_cluster_yuelpocket_predictions(pred_txt_path)
        if not preds:
            continue
            
        # Top 1 distances
        top1_pocket = preds[0]
        top1_dca = calculate_dca(top1_pocket['center'], ligand_atoms)
        top1_dcc = calculate_dcc(top1_pocket['center'], ligand_center)
        
        all_dca_top1.append(top1_dca)
        all_dcc_top1.append(top1_dcc)
        
        # Top 3 distances
        top3_preds = preds[:3]
        top3_dcas = [calculate_dca(p['center'], ligand_atoms) for p in top3_preds]
        top3_dccs = [calculate_dcc(p['center'], ligand_center) for p in top3_preds]
        
        all_dca_top3.append(min(top3_dcas))
        all_dcc_top3.append(min(top3_dccs))
        
        print(f"{pdb_id:<10} {parts[1]:<10} {top1_dca:<10.2f} {top1_dcc:<10.2f}")
        total_cases += 1
            
    if total_cases > 0:
        print("-" * 55)
        print(f"Total Cases: {total_cases}")
        
        dca_top1_rates = []
        dca_top3_rates = []
        dcc_top1_rates = []
        dcc_top3_rates = []
        
        print(f"\n{'Cutoff':<10} {'DCA Top1':<10} {'DCA Top3':<10} {'DCC Top1':<10} {'DCC Top3':<10}")
        print("-" * 55)
        
        for c in cutoffs:
            r1_dca = sum(1 for d in all_dca_top1 if d < c) / total_cases
            r3_dca = sum(1 for d in all_dca_top3 if d < c) / total_cases
            r1_dcc = sum(1 for d in all_dcc_top1 if d < c) / total_cases
            r3_dcc = sum(1 for d in all_dcc_top3 if d < c) / total_cases
            
            dca_top1_rates.append(r1_dca)
            dca_top3_rates.append(r3_dca)
            dcc_top1_rates.append(r1_dcc)
            dcc_top3_rates.append(r3_dcc)
            
            print(f"{c:<10} {r1_dca:<10.2%} {r3_dca:<10.2%} {r1_dcc:<10.2%} {r3_dcc:<10.2%}")
        
        # Plotting
        plot_success_rates(cutoffs, dca_top1_rates, dca_top3_rates, 
                          f'YuelPocket ({mode}) Success Rate (DCA)', 'Success Rate (DCA < Cutoff)', 
                          base_dir / f'yuelpocket_{mode}_{dataset_folder}_dca_success.png')
        
        plot_success_rates(cutoffs, dcc_top1_rates, dcc_top3_rates, 
                          f'YuelPocket ({mode}) Success Rate (DCC)', 'Success Rate (DCC < Cutoff)', 
                          base_dir / f'yuelpocket_{mode}_{dataset_folder}_dcc_success.png')
        
        # Save results to CSV
        csv_filename = base_dir / f'yuelpocket_{mode}_{dataset_folder}_results.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Cutoff', 'DCA_Top1', 'DCA_Top3', 'DCC_Top1', 'DCC_Top3'])
            for i, c in enumerate(cutoffs):
                writer.writerow([
                    c, 
                    f"{dca_top1_rates[i]:.4f}", 
                    f"{dca_top3_rates[i]:.4f}", 
                    f"{dcc_top1_rates[i]:.4f}", 
                    f"{dcc_top3_rates[i]:.4f}"
                ])
        print(f"Success rates saved to {csv_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate YuelPocket predictions")
    parser.add_argument("mode", help="Model mode (e.g., pos_aa3, pos_sc3)")
    parser.add_argument("dataset_folder", nargs="?", default="all", help="Dataset folder (default: all)")
    args = parser.parse_args()
    
    evaluate_predictions(args.dataset_folder, args.mode)
