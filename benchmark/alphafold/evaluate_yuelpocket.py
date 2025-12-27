import os
import csv
import sys
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Add project root to path to import app logic
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.clustering import hill_climbing_cluster

# Parameter for hill-climbing clustering
K_NN = 30

def calculate_center(atoms):
    """Calculate the geometric center (centroid) of a list of atoms."""
    if isinstance(atoms, np.ndarray):
        return np.mean(atoms, axis=0)
    coords = [atom.get_coord() for atom in atoms]
    if not coords:
        return None
    return np.mean(coords, axis=0)

def calculate_dca(pocket_center, ligand_atoms):
    """
    Calculate Distance to Center of Active site (DCA).
    Distance from pocket center to the closest ligand atom.
    """
    if pocket_center is None or (ligand_atoms is None or len(ligand_atoms) == 0):
        return float('inf')
    
    if isinstance(ligand_atoms, np.ndarray):
        dists = np.linalg.norm(ligand_atoms - pocket_center, axis=1)
        return np.min(dists)
    
    min_dist = float('inf')
    for atom in ligand_atoms:
        dist = np.linalg.norm(pocket_center - atom.get_coord())
        if dist < min_dist:
            min_dist = dist
    return min_dist

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

def get_ligand_data(lig_file):
    """Load ligand and return (atoms/coords, center)"""
    if lig_file.suffix == '.sdf':
        if not HAS_RDKIT:
            print("Error: RDKit required for .sdf files")
            return None, None
        try:
            suppl = Chem.SDMolSupplier(str(lig_file), sanitize=False)
            mol = suppl[0]
            if mol is None: return None, None
            conf = mol.GetConformer()
            coords = conf.GetPositions()
            return coords, np.mean(coords, axis=0)
        except Exception as e:
            print(f"Error loading SDF {lig_file}: {e}")
            return None, None
    elif lig_file.suffix == '.pdb':
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('ligand', str(lig_file))
            atoms = list(structure.get_atoms())
            if not atoms: return None, None
            return atoms, calculate_center(atoms)
        except Exception as e:
            print(f"Error loading PDB {lig_file}: {e}")
            return None, None
    return None, None

def plot_success_rates(cutoffs, top1_rates, top3_rates, title, ylabel, filename):
    plt.figure(figsize=(2.5, 2))
    # Convert rates to percentages
    plt.plot(cutoffs, [r * 100 for r in top1_rates], marker='o', markersize=3, 
             label='Top 1', color='#EF767B', linewidth=1.2)
    plt.plot(cutoffs, [r * 100 for r in top3_rates], marker='o', markersize=3, 
             label='Top 3', color='#43A3EF', linewidth=1.2)
    
    # Ignore passed title/ylabel to ensure consistency with other scripts
    # or just use standard ones
    plt.xlabel('Distance Threshold (Å)', fontsize=7)
    plt.ylabel('Success Rate (%)', fontsize=7)
    
    plt.xticks(cutoffs, fontsize=6)
    plt.yticks(range(0, 101, 20), fontsize=6)
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    plt.legend(fontsize=6, loc='lower right', frameon=True)
    plt.tight_layout()
    
    # Save as PNG and SVG
    plt.savefig(filename, dpi=300)
    svg_filename = str(filename).replace('.png', '.svg')
    plt.savefig(svg_filename, dpi=300)
    print(f"Plot saved to {filename} and {svg_filename}")
    plt.close()

def evaluate_predictions(dataset_folder, yuel_dir=None, mode="pos_aa3"):
    base_dir = Path(__file__).parent
    
    dataset_dir = Path(dataset_folder)
    if not dataset_dir.is_absolute():
        dataset_dir = base_dir / dataset_folder
                                
    if yuel_dir is None:
        yuel_dir = base_dir / f"{dataset_dir.name}_yuelpocket_predictions"
    else:
        yuel_dir = Path(yuel_dir)
        if not yuel_dir.is_absolute():
            yuel_dir = base_dir / yuel_dir
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    # Look for both .pdb and .sdf ligands
    ligand_files = list(dataset_dir.glob('*_ligand.pdb')) + list(dataset_dir.glob('*_ligand.sdf'))
    if not ligand_files:
        print("No ligand files found.")
        return

    cutoffs = [4, 5, 6, 7, 8, 9, 10]
    all_dca_top1 = []
    all_dca_top3 = []
    all_dcc_top1 = []
    all_dcc_top3 = []
    
    print(f"{'SystemID':<35} {'Top1_DCA':<10} {'Top1_DCC':<10}")
    print("-" * 55)

    total_cases = 0

    for lig_file in ligand_files:
        # Determine system_id
        if "_ligand" in lig_file.name:
            system_id = lig_file.name.split('_ligand')[0]
        else:
            system_id = lig_file.stem.replace('_ligand', '')
            
        prot_filename = f"{system_id}_protein.pdb"
        pred_txt_name = f"{prot_filename}_predictions.txt"
        pred_txt_path = yuel_dir / pred_txt_name
        
        if not pred_txt_path.exists():
            continue

        # Load ligand
        ligand_atoms, ligand_center = get_ligand_data(lig_file)
        if ligand_atoms is None:
            continue
            
        # Load and Cluster
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
        
        print(f"{system_id:<35} {top1_dca:<10.2f} {top1_dcc:<10.2f}")
        total_cases += 1
            
    if total_cases > 0:
        print("-" * 55)
        print(f"Total Cases: {total_cases}")
        
        # Calculate success rates for each cutoff
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
    parser = argparse.ArgumentParser(description="Evaluate YuelPocket predictions on Plinder/Holo4k datasets")
    parser.add_argument("dataset_folder", nargs="?", default="test50", help="Dataset folder (default: test50)")
    parser.add_argument("mode", nargs="?", default="pos_aa3", help="Model mode (e.g., pos_aa3, pos_sc)")
    args = parser.parse_args()
    
    pred_folder = f"{args.dataset_folder}_yuelpocket_{args.mode}_predictions"
    
    evaluate_predictions(args.dataset_folder, pred_folder, args.mode)
