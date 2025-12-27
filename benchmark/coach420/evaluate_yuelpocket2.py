import os
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt

# Add project root to path
# Assuming this script is at benchmark/coach420/evaluate_yuelpocket2.py
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import const, pdb_utils

def calculate_center(coords):
    if len(coords) == 0: return None
    return np.mean(coords, axis=0)

def calculate_dca(pocket_center, ligand_coords):
    if pocket_center is None or ligand_coords is None or len(ligand_coords) == 0:
        return float('inf')
    dists = np.linalg.norm(ligand_coords - pocket_center, axis=1)
    return np.min(dists)

def calculate_dcc(pocket_center, ligand_center):
    if pocket_center is None or ligand_center is None:
        return float('inf')
    return np.linalg.norm(pocket_center - ligand_center)

def load_ligand_data(lig_file):
    if lig_file.suffix == '.sdf':
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(lig_file), sanitize=False)
        mol = suppl[0]
        if mol is None: return None, None
        coords = mol.GetConformer().GetPositions()
        return coords, np.mean(coords, axis=0)
    elif lig_file.suffix == '.pdb':
        structure = pdb_utils.Structure(str(lig_file))
        atoms = structure.get_atoms()
        coords = np.array([atom.get_coord() for atom in atoms])
        if len(coords) == 0: return None, None
        return coords, np.mean(coords, axis=0)
    return None, None

def load_yuelpocket_clusters(csv_path):
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
        if df.empty: return []
        clusters = []
        for cid, group in df.groupby('ClusterID'):
            center = group[['X', 'Y', 'Z']].values.mean(axis=0)
            score = group['Score'].sum()
            clusters.append({
                'id': cid,
                'center': center,
                'score': score
            })
        clusters.sort(key=lambda x: x['score'], reverse=True)
        return clusters
    except Exception as e:
        print(f"Error loading clusters from {csv_path}: {e}")
        return []

def plot_success_rates(cutoffs, top1_rates, top3_rates, ylabel, filename):
    plt.figure(figsize=(3, 2.5))
    plt.plot(cutoffs, [r*100 for r in top1_rates], marker='o', markersize=3, 
             label='Top 1', color='#EF767B', linewidth=1.2)
    plt.plot(cutoffs, [r*100 for r in top3_rates], marker='o', markersize=3, 
             label='Top 3', color='#43A3EF', linewidth=1.2)
    
    plt.xlabel('Distance Threshold (Å)', fontsize=7)
    plt.ylabel(ylabel, fontsize=7)
    plt.xticks(cutoffs, fontsize=6)
    plt.yticks(range(0, 101, 20), fontsize=6)
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    plt.legend(fontsize=6, loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(str(filename).replace('.png', '.svg'), dpi=300)
    plt.close()

def evaluate_residues_and_pockets(dataset_folder, yuel_dir=None):
    base_dir = Path(__file__).parent
    dataset_dir = Path(dataset_folder)
    if not dataset_dir.is_absolute():
        dataset_dir = base_dir / dataset_folder
        
    if yuel_dir is None:
        yuel_dir = base_dir / f"{dataset_dir.name}_yuelpocket_residues2_predictions"
    else:
        yuel_dir = Path(yuel_dir)
        if not yuel_dir.is_absolute():
            yuel_dir = base_dir / yuel_dir

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    ligand_files = list(dataset_dir.glob('*_ligand.sdf')) + list(dataset_dir.glob('*_ligand.pdb'))
    if not ligand_files:
        ligand_files = list(dataset_dir.glob('*ligand.pdb'))

    if not ligand_files:
        print("No ligand files found.")
        return

    # Metrics
    all_y_true = []
    all_y_pred_prob = []
    residue_results = []
    
    cutoffs = [4, 5, 6, 7, 8, 9, 10]
    all_dca_top1 = []
    all_dca_top3 = []
    all_dcc_top1 = []
    all_dcc_top3 = []
    all_system_residue_dist_scores = []
    
    print(f"{'SystemID':<35} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Top1_DCA':<10} {'BestPktRank':<12}")
    print("-" * 105)

    for lig_file in ligand_files:
        if "_ligand" in lig_file.name:
            system_id = lig_file.name.split('_ligand')[0]
            prot_file = dataset_dir / f"{system_id}_protein.pdb"
        else:
            system_id = lig_file.name.replace('_ligand.sdf', '').replace('_ligand.pdb', '').replace('ligand.pdb', '')
            prot_file = dataset_dir / f"{system_id}_protein.pdb"

        pred_base = yuel_dir / f"{system_id}_predictions"
        pred_txt = Path(str(pred_base) + ".txt")
        cluster_csv = Path(str(pred_base) + "_clusters.csv")
        
        if not prot_file.exists():
            prot_file = dataset_dir / f"{system_id}.pdb"
        if not prot_file.exists(): continue
        
        lig_coords, lig_center = load_ligand_data(lig_file)
        if lig_coords is None: continue

        case_p, case_r, case_f = 0.0, 0.0, 0.0
        case_best_rank = float('nan')
        
        try:
            structure = pdb_utils.Structure(str(prot_file), skip_hetatm=True)
            all_ligand_coords, _ = load_ligand_data(lig_file)
            lig_tree = KDTree(all_ligand_coords)

            pocket_residue_keys = set()
            all_residues = []
            
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
                break
            
            pred_scores = {}
            if pred_txt.exists():
                preds_df = pd.read_csv(pred_txt, comment='#')
                for _, row in preds_df.iterrows():
                    key = (row['Chain'] if not pd.isna(row['Chain']) else ' ', int(row['ResID']))
                    pred_scores[key] = row['PocketProbability']
            
            if all_residues:
                system_data = []
                for res in all_residues:
                    score = pred_scores.get(res['key'], 0.0)
                    system_data.append((res['min_dist'], score))
                all_system_residue_dist_scores.append(system_data)
                
                residue_data_6A = [(d < 6.0, s) for d, s in system_data]
                residue_data_6A.sort(key=lambda x: x[1], reverse=True)
                
                for rank, (is_pocket, _) in enumerate(residue_data_6A, start=1):
                    if is_pocket:
                        case_best_rank = rank
                        break
                
                y_true_subset = []
                y_score_subset = []
                for key, score in pred_scores.items():
                    is_pocket = 1 if key in pocket_residue_keys else 0
                    y_true_subset.append(is_pocket)
                    y_score_subset.append(score)
                
                if y_true_subset and sum(y_true_subset) > 0:
                    y_true_subset = np.array(y_true_subset)
                    y_score_subset = np.array(y_score_subset)
                    y_pred_subset = (y_score_subset > 0.5).astype(int)
                    case_p, case_r, case_f, _ = precision_recall_fscore_support(y_true_subset, y_pred_subset, average='binary', zero_division=0)
                    all_y_true.extend(y_true_subset)
                    all_y_pred_prob.extend(y_score_subset)
                
                residue_results.append({
                    'SystemID': system_id, 'Precision': case_p, 'Recall': case_r, 
                    'F1': case_f, 'BestPktRank': case_best_rank
                })

        except Exception as e:
            print(f"Residue eval failed for {system_id}: {e}")

        # Pocket Metrics
        case_top1_dca = float('inf')
        if cluster_csv.exists():
            clusters = load_yuelpocket_clusters(cluster_csv)
            if clusters:
                top1_pocket = clusters[0]
                case_top1_dca = calculate_dca(top1_pocket['center'], lig_coords)
                case_top1_dcc = calculate_dcc(top1_pocket['center'], lig_center)
                all_dca_top1.append(case_top1_dca)
                all_dcc_top1.append(case_top1_dcc)
                
                top3_clusters = clusters[:3]
                top3_dcas = [calculate_dca(c['center'], lig_coords) for c in top3_clusters]
                top3_dccs = [calculate_dcc(c['center'], lig_center) for c in top3_clusters]
                all_dca_top3.append(min(top3_dcas))
                all_dcc_top3.append(min(top3_dccs))
        
        print(f"{system_id:<35} {case_p:<10.4f} {case_r:<10.4f} {case_f:<10.4f} {case_top1_dca:<10.2f} {case_best_rank:<12.0f}")

    # Summaries
    if all_y_true:
        all_y_true = np.array(all_y_true)
        all_y_pred_prob = np.array(all_y_pred_prob)
        all_y_pred = (all_y_pred_prob > 0.5).astype(int)
        avg_p, avg_r, avg_f, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary', zero_division=0)
        global_auc = roc_auc_score(all_y_true, all_y_pred_prob)
        print("-" * 105)
        print(f"RESIDUE GLOBAL: Precision={avg_p:.4f}, Recall={avg_r:.4f}, F1={avg_f:.4f}, AUC={global_auc:.4f}")

    if all_system_residue_dist_scores:
        dist_thresholds = [4, 5, 6, 7, 8, 9, 10]
        rank_ks = [1, 3, 10]
        n_systems = len(all_system_residue_dist_scores)
        success_matrix = np.zeros((len(rank_ks), len(dist_thresholds)))
        
        for d_idx, d_thresh in enumerate(dist_thresholds):
            system_ranks_at_d = []
            for system_data in all_system_residue_dist_scores:
                sorted_res = sorted(system_data, key=lambda x: x[1], reverse=True)
                best_r = float('inf')
                for rank, (dist, score) in enumerate(sorted_res, start=1):
                    if dist < d_thresh:
                        best_r = rank
                        break
                system_ranks_at_d.append(best_r)
            for k_idx, k in enumerate(rank_ks):
                success_matrix[k_idx, d_idx] = sum(1 for r in system_ranks_at_d if r <= k) / n_systems * 100

        colors = ['#EF767B', '#43A3EF', 'black']
        plt.figure(figsize=(3, 2.5))
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
        plt.savefig(base_dir / f'evaluate_residues2_{dataset_dir.name}_rank_curves.svg', dpi=300)
        plt.close()
        
        print(f"\nRANK SUCCESS RATES (%):")
        header = f"{'Rank K':<10}" + "".join([f"{d:>8}A" for d in dist_thresholds])
        print(header)
        for k_idx, k in enumerate(rank_ks):
            print(f"{k:<10}" + "".join([f"{success_matrix[k_idx, d_idx]:>9.1f}%" for d_idx in range(len(dist_thresholds))]))

    if all_dca_top1:
        total_cases = len(all_dca_top1)
        print(f"\nPOCKET SUCCESS RATES (Total {total_cases}):")
        print(f"{'Cutoff':<10} {'DCA Top1':<10} {'DCA Top3':<10} {'DCC Top1':<10} {'DCC Top3':<10}")
        print("-" * 60)
        
        dca_top1_rates, dca_top3_rates = [], []
        dcc_top1_rates, dcc_top3_rates = [], []
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

        plot_success_rates(cutoffs, dca_top1_rates, dca_top3_rates, 
                          'Success Rate (%)', 
                          base_dir / f'evaluate_residues2_{dataset_dir.name}_dca.png')
        plot_success_rates(cutoffs, dcc_top1_rates, dcc_top3_rates, 
                          'Success Rate (%)', 
                          base_dir / f'evaluate_residues2_{dataset_dir.name}_dcc.png')

    if residue_results:
        pd.DataFrame(residue_results).to_csv(base_dir / f"evaluate_residues2_{dataset_dir.name}_residues.csv", index=False)
    if all_dca_top1:
         with open(base_dir / f"evaluate_residues2_{dataset_dir.name}_pockets.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Cutoff', 'DCA_Top1', 'DCA_Top3', 'DCC_Top1', 'DCC_Top3'])
            for i, c in enumerate(cutoffs):
                writer.writerow([c, dca_top1_rates[i], dca_top3_rates[i], dcc_top1_rates[i], dcc_top3_rates[i]])

    print(f"\nResults saved to CSV files in {base_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", nargs="?", default="test_removed_notplinder")
    parser.add_argument("--yuel_dir", help="Directory with predictions")
    args = parser.parse_args()
    evaluate_residues_and_pockets(args.dataset_folder, args.yuel_dir)
