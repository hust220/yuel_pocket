#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../../')
from src.db_utils import db_connection
from tqdm import tqdm
import random
from collections import defaultdict
from io import StringIO
from src.db_utils import db_select
from src.clustering import write_bfactor_to_pdb

def get_all_proteins():
    """Get all protein names that have probe predictions in probe_moad_test."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT protein_name FROM probe_moad_test")
        return [row[0] for row in cur.fetchall()]

def get_all_probes():
    """Get all probe names in a fixed order from probe_moad_test."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT probe_name FROM probe_moad_test ORDER BY probe_name")
        return [row[0] for row in cur.fetchall()]

def get_all_probe_predictions(pname):
    """Get pocket predictions for all probes of a specific protein from probe_moad_test."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT probe_name, pocket_pred
            FROM probe_moad_test
            WHERE protein_name = %s
        """, (pname,))
        results = {}
        for row in cur.fetchall():
            probe_name, pocket_pred = row
            results[probe_name] = pickle.loads(pocket_pred)
        return results

def get_is_pocket_union(pname):
    """获取该蛋白所有ligand的is_pocket并集，并返回实际长度"""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT lname FROM processed_datasets WHERE pname = %s
        """, (pname,))
        ligands = [row[0] for row in cur.fetchall()]
        pocket_union = None
        maxlen = 0
        for lname in ligands:
            cur.execute("SELECT is_pocket FROM processed_datasets WHERE lname = %s", (lname,))
            result = cur.fetchone()
            if result is None:
                continue
            is_pocket = pickle.loads(result[0])
            maxlen = max(maxlen, len(is_pocket))
            if pocket_union is None:
                pocket_union = (is_pocket > 0).astype(np.int32)
            else:
                minlen = min(len(pocket_union), len(is_pocket))
                pocket_union = np.logical_or(
                    pocket_union[:minlen], (is_pocket[:minlen] > 0)
                ).astype(np.int32)
        return pocket_union, maxlen

def calculate_metrics():
    proteins = get_all_proteins()
    probes = get_all_probes()
    n_probe_max = len(probes)
    all_recall = {n: [] for n in range(1, n_probe_max+1)}
    all_precision = {n: [] for n in range(1, n_probe_max+1)}
    all_positive_ratio = {n: [] for n in range(1, n_probe_max+1)}
    print(f"Total proteins: {len(proteins)}, probes: {n_probe_max}")
    for pname in tqdm(proteins, desc="Proteins"):
        probe_preds = get_all_probe_predictions(pname)
        if len(probe_preds) == 0:
            continue
        is_pocket, _ = get_is_pocket_union(pname)
        if is_pocket is None or np.sum(is_pocket) == 0:
            continue
        probe_names = [p for p in probes if p in probe_preds]
        if len(probe_names) == 0:
            continue
        pred_matrix = np.stack([probe_preds[p] for p in probe_names], axis=0)
        minlen = min(pred_matrix.shape[1], len(is_pocket))
        is_pocket = is_pocket[:minlen]
        for n in range(1, min(len(probe_names), n_probe_max)+1):
            # IQR方法合并掩码
            combined_pred_mask = np.zeros(minlen, dtype=bool)
            for i in range(n):
                pred = pred_matrix[i, :minlen]
                q1 = np.percentile(pred, 25)
                q3 = np.percentile(pred, 75)
                iqr = q3 - q1
                iqr_threshold = q3 + 1.5 * iqr
                combined_pred_mask = combined_pred_mask | (pred > iqr_threshold)
            # recall: 真实pocket中被预测为pocket的比例
            mask = (is_pocket == 1)
            if np.sum(mask) == 0:
                continue
            recall = np.mean(combined_pred_mask[mask])
            all_recall[n].append(recall)
            # precision: 预测为pocket中真实为pocket的比例
            if np.sum(combined_pred_mask) == 0:
                precision = np.nan
            else:
                precision = np.mean(is_pocket[combined_pred_mask])
            all_precision[n].append(precision)
            # positive ratio: 预测为pocket的残基比例
            positive_ratio = np.mean(combined_pred_mask)
            all_positive_ratio[n].append(positive_ratio)
    return all_recall, all_precision, all_positive_ratio

def plot_recall(all_ratios):
    import matplotlib.cm as cm
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(3.5, 2.5))
    n_lines = len(all_ratios)
    cmap = cm.get_cmap('Reds', n_lines+2)  # 红色渐变
    bins = np.linspace(0, 1, 21)
    sorted_items = sorted(all_ratios.items())
    min_idx = 0
    max_idx = n_lines - 1
    for idx, (n, ratios) in enumerate(sorted_items):
        if len(ratios) == 0:
            continue
        ratios = np.array(ratios)
        y = [(ratios > t).mean() for t in bins]
        color = cmap(idx + 2)
        if idx == min_idx:
            plt.plot(bins, y, color=color, linewidth=1.5, label=f'n={n}')
        elif idx == max_idx:
            plt.plot(bins, y, color=color, linewidth=1.5, label=f'n={n}')
        else:
            plt.plot(bins, y, color=color, linewidth=1.5)
    plt.xlabel('Ratio of Predicted Pocket Residues')
    plt.ylabel('Fraction of proteins (> ratio)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=False)
    filename = 'plots/probe_success_ratio_all.svg'
    print(filename)
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_precision(all_ratios):
    import matplotlib.cm as cm
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(3.5, 2.5))
    n_lines = len(all_ratios)
    cmap = cm.get_cmap('Reds', n_lines+2)
    bins = np.linspace(0, 1, 21)
    sorted_items = sorted(all_ratios.items())
    min_idx = 0
    max_idx = n_lines - 1
    for idx, (n, ratios) in enumerate(sorted_items):
        if len(ratios) == 0:
            continue
        ratios = np.array(ratios)
        # precision曲线：每个阈值下大于该ratio的蛋白数/总蛋白数
        y = [(ratios > t).sum() / len(ratios) for t in bins]
        color = cmap(idx + 2)
        if idx == min_idx:
            plt.plot(bins, y, color=color, linewidth=1.5, label=f'n={n}')
        elif idx == max_idx:
            plt.plot(bins, y, color=color, linewidth=1.5, label=f'n={n}')
        else:
            plt.plot(bins, y, color=color, linewidth=1.5)
    plt.xlabel('Ratio of Predicted Pocket Residues')
    plt.ylabel('Precision (> ratio)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=False)
    filename = 'plots/probe_success_precision_all.svg'
    print(filename)
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_combined_pred_positive_ratio(all_positive_ratio):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    os.makedirs('plots', exist_ok=True)
    n_list = sorted(all_positive_ratio.keys())
    means = [np.nanmean(all_positive_ratio[n]) for n in n_list]
    stds = [np.nanstd(all_positive_ratio[n]) for n in n_list]
    cmap = cm.get_cmap('Reds', len(n_list)+2)
    color = cmap(len(n_list))
    # 计算所有蛋白is_pocket为真的平均比例和std
    proteins = get_all_proteins()
    is_pocket_ratios = []
    for pname in proteins:
        is_pocket, _ = get_is_pocket_union(pname)
        if is_pocket is not None and len(is_pocket) > 0:
            is_pocket_ratios.append(np.mean(is_pocket))
    is_pocket_mean = np.mean(is_pocket_ratios)
    is_pocket_std = np.std(is_pocket_ratios)
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(n_list, means, color=color, linewidth=2)
    plt.fill_between(n_list, np.array(means)-np.array(stds), np.array(means)+np.array(stds), color=color, alpha=0.2)
    # 添加黑色横线和阴影
    plt.axhline(is_pocket_mean, color='black', linestyle='-', linewidth=1.5)
    plt.fill_between(n_list, is_pocket_mean-is_pocket_std, is_pocket_mean+is_pocket_std, color='black', alpha=0.15)
    plt.xlabel('Number of Probes (n)')
    plt.ylabel('Mean Fraction of Residues Predicted as Pocket')
    plt.grid(True, linestyle='--', alpha=0.5)
    filename = 'plots/combined_pred_positive_ratio_vs_n.svg'
    print(filename)
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def get_all_ligands(pname):
    """Get all ligand names for a given protein from processed_datasets."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT lname FROM processed_datasets WHERE pname = %s", (pname,))
        return [row[0] for row in cur.fetchall()]

def get_ligand_is_pocket(lname):
    """Get is_pocket array for a given ligand name."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT is_pocket FROM processed_datasets WHERE lname = %s", (lname,))
        result = cur.fetchone()
        if result is None:
            return None
        return pickle.loads(result[0])

def iqr_mask(arr):
    arr = np.array(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    iqr_threshold = q3 + 1.5 * iqr
    return arr > iqr_threshold

def compute_intersection(pocket_pred, is_pocket):
    is_pocket_mask = np.array(is_pocket) > 0
    pocket_pred_mask = iqr_mask(pocket_pred)
    minlen = min(len(pocket_pred_mask), len(is_pocket_mask))
    return np.logical_and(pocket_pred_mask[:minlen], is_pocket_mask[:minlen]).sum()

def get_all_protein_ligand_pairs():
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pname, lname FROM processed_datasets where split = 'test'")
        return cur.fetchall()  # list of (pname, lname)

def find_multi_pocket_probe_example_once(min_ligands=3, intersection_threshold=10, probe_pred_threshold=0.5, seed=None):
    pairs = get_all_protein_ligand_pairs()
    protein_to_ligands = defaultdict(list)
    for pname, lname in pairs:
        protein_to_ligands[pname].append(lname)
    proteins = list(protein_to_ligands.keys())
    if seed is not None:
        random.seed(seed)
    random.shuffle(proteins)
    for pname in tqdm(proteins, desc="Proteins"):
        ligands = protein_to_ligands[pname]
        if len(ligands) < min_ligands:
            print(f"protein {pname} has less than {min_ligands} ligands")
            continue
        ligand_pockets = {}
        maxlen = 0
        for lname in ligands:
            is_pocket = get_ligand_is_pocket(lname)
            if is_pocket is not None:
                ligand_pockets[lname] = is_pocket
                maxlen = max(maxlen, len(is_pocket))
        if len(ligand_pockets) < min_ligands:
            print(f"protein {pname} has less than {min_ligands} ligands with pocket")
            continue
        probe_preds = get_all_probe_predictions(pname)
        if len(probe_preds) == 0:
            print(f"protein {pname} has no probe predictions")
            continue
        # 对齐长度
        for lname in ligand_pockets:
            ligand_pockets[lname] = ligand_pockets[lname][:maxlen]
        probe_ligand_map = {}
        for probe in probe_preds:
            probe_pred = probe_preds[probe]
            intersections = {}
            for lname, is_pocket in ligand_pockets.items():
                minlen = min(len(probe_pred), len(is_pocket))
                intersection = compute_intersection(probe_pred[:minlen], is_pocket[:minlen])
                intersections[lname] = intersection
            best_ligand = max(intersections, key=intersections.get)
            best_intersection = intersections[best_ligand]
            if best_intersection >= intersection_threshold:
                probe_ligand_map[probe] = (best_ligand, best_intersection)
        ligands_hit = set(l for l, n in probe_ligand_map.values())
        print(len(ligands_hit))
        if len(ligands_hit) >= min_ligands:
            print("protein, probe, ligand, intersection")
            for probe, (ligand, intersection) in probe_ligand_map.items():
                print(f"{pname}, {probe}, {ligand}, {intersection}")
            return {pname: probe_ligand_map}
    print("No example found.")
    return None

def save_example_data(data, outdir="example_output"):
    import os
    os.makedirs(outdir, exist_ok=True)
    for pname, probe_ligand_map in data.items():
        protein_dir = os.path.join(outdir, pname)
        os.makedirs(protein_dir, exist_ok=True)
        ligands = set(ligand for ligand, _ in probe_ligand_map.values())
        # 保存ligand的SDF和is_pocket PDB
        for ligand in ligands:
            # 保存SDF
            with db_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT mol FROM ligands WHERE name = %s AND protein_name = %s", (ligand, pname))
                mol_data = cur.fetchone()
                if mol_data and mol_data[0] is not None:
                    with open(os.path.join(protein_dir, f"{ligand}.ligand.sdf"), 'w') as f:
                        f.write(mol_data[0].tobytes().decode('utf-8'))
            # 保存is_pocket为PDB
            is_pocket = get_ligand_is_pocket(ligand)
            pdb_content = db_select('proteins', 'pdb', f"name = '{pname}'", is_bytea=True)
            write_bfactor_to_pdb(np.array(is_pocket, dtype=float), pdb_content, output_file=os.path.join(protein_dir, f"{ligand}_is_pocket.pdb"))
        # 保存probe的pocket_pred为PDB
        for probe, (ligand, intersection) in probe_ligand_map.items():
            probe_preds = get_all_probe_predictions(pname)
            pocket_pred = probe_preds[probe]
            pdb_content = db_select('proteins', 'pdb', f"name = '{pname}'", is_bytea=True)
            write_bfactor_to_pdb(
                np.array(pocket_pred, dtype=float),
                pdb_content,
                output_file=os.path.join(protein_dir, f"{probe}_pred_for_{ligand}.pdb")
            )

# 主流程调用
if __name__ == "__main__":
    # recall_metrics, precision_metrics, positive_ratio_metrics = calculate_metrics()
    # plot_recall(recall_metrics)
    # plot_precision(precision_metrics)
    # plot_combined_pred_positive_ratio(positive_ratio_metrics)
    data = find_multi_pocket_probe_example_once()
    save_example_data(data, outdir="example_output")
# %%
