import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import const, pdb_utils

def load_ligand_data(lig_file):
    """Load ligand atom coordinates."""
    if lig_file.suffix == '.sdf':
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(lig_file), sanitize=False)
        mol = suppl[0]
        if mol is None: return None
        coords = mol.GetConformer().GetPositions()
        return coords
    elif lig_file.suffix == '.pdb':
        structure = pdb_utils.Structure(str(lig_file))
        atoms = structure.get_atoms()
        coords = np.array([atom.get_coord() for atom in atoms])
        if len(coords) == 0: return None
        return coords
    return None

def parse_pm_pdb(pdb_path):
    """Extract residue scores from PocketMiner output PDB (B-factors)."""
    scores = {}
    if not os.path.exists(pdb_path):
        return scores
        
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                # Normalize empty chain to space
                if not chain_id.strip():
                    chain_id = ' '
                res_seq = int(line[22:26].strip())
                # PocketMiner usually outputs CA or backbone scores in B-factor column [60-66]
                score = float(line[60:66].strip())
                key = (chain_id, res_seq)
                if key not in scores:
                    scores[key] = score
                else:
                    # Usually all atoms of a residue have the same score in PM
                    scores[key] = max(scores[key], score)
    return scores

def evaluate_pm(dataset_folder="test50"):
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / dataset_folder
    pm_dir = base_dir / f"{dataset_folder}_pocketminer_predictions"
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    ligand_files = list(dataset_dir.glob('*_ligand.sdf')) + list(dataset_dir.glob('*_ligand.pdb'))
    if not ligand_files:
        print("No ligand files found.")
        return

    print(f"Evaluating PocketMiner on {dataset_folder}...")
    
    all_y_true = []
    all_y_pred_prob = []
    all_system_residue_dist_scores = []
    
    print(f"{'SystemID':<35} {'Precision':<10} {'Recall':<10} {'F1':<10} {'BestPktRank':<12}")
    print("-" * 85)

    for lig_file in ligand_files:
        system_id = lig_file.name.split('_ligand')[0]
        prot_file = dataset_dir / f"{system_id}_protein.pdb"
        pred_pdb = pm_dir / f"{system_id}_predictions.pdb"
        
        if not prot_file.exists() or not pred_pdb.exists():
            continue
            
        lig_coords = load_ligand_data(lig_file)
        if lig_coords is None: continue
        lig_tree = KDTree(lig_coords)

        try:
            # 1. Parse protein and identify pocket residues
            structure = pdb_utils.Structure(str(prot_file), skip_hetatm=True)
            all_residues = []
            pocket_residue_keys = set()
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.res_name in const.ALLOWED_RESIDUE_TYPES:
                            res_coords = residue.get_coords()
                            if len(res_coords) == 0: continue
                            
                            dists, _ = lig_tree.query(res_coords)
                            min_dist = np.min(dists)
                            is_pocket = min_dist < 6.0
                            
                            key = (chain.chain_id if chain.chain_id.strip() else ' ', residue.res_id)
                            all_residues.append({'key': key, 'min_dist': min_dist, 'is_pocket': is_pocket})
                            if is_pocket:
                                pocket_residue_keys.add(key)
                break # Only first model

            # 2. Load PM scores
            pred_scores = parse_pm_pdb(pred_pdb)
            
            # 3. Evaluate Ranking (on ALL residues)
            if all_residues:
                system_data = []
                for res in all_residues:
                    score = pred_scores.get(res['key'], 0.0)
                    system_data.append((res['min_dist'], score))
                
                all_system_residue_dist_scores.append(system_data)
                
                # Best rank at 6.0A for display
                res_data_6A = sorted([(d < 6.0, s) for d, s in system_data], key=lambda x: x[1], reverse=True)
                case_best_rank = float('nan')
                for rank, (is_pocket, _) in enumerate(res_data_6A, start=1):
                    if is_pocket:
                        case_best_rank = rank
                        break

                # 4. Precision/Recall on predicted subset
                y_true_subset = []
                y_score_subset = []
                for key, score in pred_scores.items():
                    is_pocket = 1 if key in pocket_residue_keys else 0
                    y_true_subset.append(is_pocket)
                    y_score_subset.append(score)
                
                case_p, case_r, case_f = 0.0, 0.0, 0.0
                if y_true_subset and sum(y_true_subset) > 0:
                    y_true_subset = np.array(y_true_subset)
                    y_score_subset = np.array(y_score_subset)
                    # PM doesn't have a standard threshold, use 0.5 or mean? 
                    # Many baseline papers use 0.5 for classification metrics.
                    y_pred_subset = (y_score_subset > 0.5).astype(int)
                    case_p, case_r, case_f, _ = precision_recall_fscore_support(y_true_subset, y_pred_subset, average='binary', zero_division=0)
                    all_y_true.extend(y_true_subset)
                    all_y_pred_prob.extend(y_score_subset)
                
                print(f"{system_id:<35} {case_p:<10.4f} {case_r:<10.4f} {case_f:<10.4f} {case_best_rank:<12.0f}")

        except Exception as e:
            print(f"Eval failed for {system_id}: {e}")

    # Summary Plots - reusing the style from YuelPocket
    if all_system_residue_dist_scores:
        dist_thresholds = [4, 5, 6, 7, 8, 9, 10]
        rank_ks = [1, 3, 10]
        n_systems = len(all_system_residue_dist_scores)
        success_matrix = np.zeros((len(rank_ks), len(dist_thresholds)))
        
        for d_idx, d_thresh in enumerate(dist_thresholds):
            system_ranks = []
            for sys_data in all_system_residue_dist_scores:
                sorted_res = sorted(sys_data, key=lambda x: x[1], reverse=True)
                best_r = float('inf')
                for rank, (dist, score) in enumerate(sorted_res, start=1):
                    if dist < d_thresh:
                        best_r = rank
                        break
                system_ranks.append(best_r)
            
            for k_idx, k in enumerate(rank_ks):
                success_matrix[k_idx, d_idx] = sum(1 for r in system_ranks if r <= k) / n_systems * 100

        # Plot
        colors = ['#EF767B', '#43A3EF', 'black']
        plt.figure(figsize=(3.5, 2.5))
        for k_idx, k in enumerate(rank_ks):
            plt.plot(dist_thresholds, success_matrix[k_idx, :], marker='o', markersize=3, 
                     label=fr'Rank $\leq$ {k}', color=colors[k_idx], linewidth=1.2)
        
        plt.xlabel('Distance Threshold (Å)', fontsize=7)
        plt.ylabel('Success Rate (%)', fontsize=7)
        plt.xticks(dist_thresholds, fontsize=6)
        plt.yticks(range(0, 101, 20), fontsize=6)
        plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
        plt.legend(fontsize=6, loc='lower right', frameon=True)
        plt.tight_layout()
        plt.savefig(base_dir / f'evaluate_pm_{dataset_folder}_rank_curves.png', dpi=300)
        plt.close()

        print(f"\nPocketMiner Rank Success Summary:")
        header = f"{'Rank K':<10}" + "".join([f"{d:>8}A" for d in dist_thresholds])
        print(header)
        for k_idx, k in enumerate(rank_ks):
            row = f"{k:<10}" + "".join([f"{success_matrix[k_idx, d_idx]:>9.1f}%" for d_idx in range(len(dist_thresholds))])
            print(row)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default="test50")
    args = parser.parse_args()
    evaluate_pm(args.folder)
