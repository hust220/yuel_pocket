#%%

import pickle
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../..')
from src.db_utils import db_connection

def calc_top3_cluster_mean(pocket_pred, clusters):
    """计算top 3 clusters的平均概率（先对每个cluster取均值，再排序，最后合并top3 cluster的残基算均值）"""
    pocket_pred = np.asarray(pocket_pred).squeeze()
    cluster_labels = np.asarray(clusters)
    min_length = min(len(pocket_pred), len(cluster_labels))
    pocket_pred = pocket_pred[:min_length]
    cluster_labels = cluster_labels[:min_length]
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]
    if len(unique_clusters) == 0:
        return None
    cluster_means = []
    for cluster_id in unique_clusters:
        cluster_mask = (cluster_labels == cluster_id)
        cluster_probs = pocket_pred[cluster_mask]
        cluster_mean = np.mean(cluster_probs)
        cluster_means.append((cluster_id, cluster_mean))
    # 按均值排序，取top3
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    top_clusters = [cid for cid, _ in cluster_means[:3]]
    mask = np.isin(cluster_labels, top_clusters)
    if np.sum(mask) == 0:
        return None
    return float(np.mean(pocket_pred[mask]))

def calc_pred_above_005_mean(pocket_pred, clusters=None):
    """计算pocket_pred中大于0.05的所有位置的平均值作为分数"""
    pocket_pred = np.asarray(pocket_pred).squeeze()
    mask = pocket_pred > 0.05
    if np.sum(mask) == 0:
        return None
    return float(np.mean(pocket_pred[mask]))

def calc_pred_max(pocket_pred, clusters=None):
    """选择pocket_pred中的最大概率作为分数"""
    pocket_pred = np.asarray(pocket_pred).squeeze()
    if pocket_pred.size == 0:
        return None
    return float(np.max(pocket_pred))

def get_screening_power_results():
    with db_connection() as conn:
        cur = conn.cursor()
        # 获取所有target
        cur.execute("SELECT DISTINCT target FROM casf_predictions")
        targets = [row[0] for row in cur.fetchall()]
        print(f"Total targets: {len(targets)}")
        top_percentages = [0.01, 0.05, 0.10]
        success_counts = {p: 0 for p in top_percentages}
        ef_sums = {p: 0.0 for p in top_percentages}
        n_targets = 0
        for target in tqdm(targets, desc="Analyzing targets"):
            # 取出该target所有ligand及分数和ligand_rank
            cur.execute("""
                SELECT p.ligand, p.pocket_pred, p.clusters, s.ligand_rank
                FROM casf_predictions p
                LEFT JOIN casf_screening s
                ON p.target = s.target AND p.ligand = s.ligand
                WHERE p.target = %s
            """, (target,))
            rows = cur.fetchall()
            ligands, scores, ligand_ranks = [], [], []
            for ligand, pocket_pred, clusters, ligand_rank in rows:
                if pocket_pred is None or clusters is None:
                    continue
                pocket_pred = pickle.loads(pocket_pred)
                clusters = pickle.loads(clusters)
                # score = calc_top3_cluster_mean(pocket_pred, clusters)
                # score = calc_pred_above_005_mean(pocket_pred, clusters)
                score = calc_pred_max(pocket_pred, clusters)
                if score is not None:
                    ligands.append(ligand)
                    scores.append(score)
                    ligand_ranks.append(ligand_rank)
            # if len(scores) < 285:
            #     continue  # 跳过不完整的target
            # 排序
            sorted_idx = np.argsort(scores)[::-1]
            ligands = np.array(ligands)[sorted_idx]
            scores = np.array(scores)[sorted_idx]
            ligand_ranks = np.array(ligand_ranks)[sorted_idx]
            n_total = len(scores)
            # 找到ligand_rank为1的ligand在排序后的排名
            rank1_idx = None
            for idx, r in enumerate(ligand_ranks):
                if r == 1:
                    rank1_idx = idx
                    break
            if rank1_idx is None:
                continue  # 没有rank1 ligand，跳过
            n_targets += 1
            for p in top_percentages:
                top_n = max(1, int(np.ceil(n_total * p)))
                # 只有rank1 ligand在top_n内才算success
                success = 1 if rank1_idx < top_n else 0
                # EF依然统计top_n内所有true binder
                top_true = np.sum([r is not None and r <= 5 for r in ligand_ranks[:top_n]])
                n_true = np.sum([r is not None and r <= 5 for r in ligand_ranks])
                ef = (top_true / top_n) / (n_true / n_total) if n_true > 0 else 0
                success_counts[p] += success
                ef_sums[p] += ef
        if n_targets == 0:
            print("No valid targets analyzed. Please check your data and clustering results.")
            return
        print(f"Total targets analyzed: {n_targets}")
        for p in top_percentages:
            print(f"Top {int(p*100)}%:")
            print(f"  Success Rate: {success_counts[p] / n_targets:.3f}")
            print(f"  Enhancement Factor (EF): {ef_sums[p] / n_targets:.3f}")

def plot_random_targets_swarm_plot(n_targets=3, save_path=None):
    """Plot swarm plots of ligand scores for randomly selected targets.
    
    Args:
        n_targets: Number of random targets to plot
        save_path: Path to save the plot. If None, plot will be shown instead.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import random
    import pickle
    
    # Get all unique targets
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT target FROM casf_predictions")
        all_targets = [row[0] for row in cur.fetchall()]
    
    # Randomly select n_targets
    selected_targets = random.sample(all_targets, min(n_targets, len(all_targets)))
    
    # Get data for selected targets
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.target, p.ligand, p.pocket_pred, p.clusters, s.ligand_rank
            FROM casf_predictions p
            LEFT JOIN casf_screening s
            ON p.target = s.target AND p.ligand = s.ligand
            WHERE p.target = ANY(%s)
        """, (selected_targets,))
        results = cur.fetchall()
    
    # Prepare data for plotting
    data = []
    for target, ligand, pocket_pred, clusters, ligand_rank in results:
        if pocket_pred is None or clusters is None:
            continue
        pocket_pred = pickle.loads(pocket_pred)
        clusters = pickle.loads(clusters)
        score = calc_pred_max(pocket_pred, clusters)
        if score is not None:
            data.append({
                'Target': target,
                'Score': score,
                'Is Top Rank': ligand_rank == 1
            })
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create swarm plot
    ax = sns.swarmplot(data=df, x='Target', y='Score', hue='Is Top Rank', 
                      palette={True: 'red', False: 'blue'}, size=8)
    
    # Customize plot
    plt.title('Ligand Scores Distribution for Random Targets', pad=20)
    plt.xlabel('Target PDB')
    plt.ylabel('Score (Max Probability)')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, ['Other Ligands', 'Top Rank Ligand'], 
              title='Ligand Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # get_screening_power_results()
    plot_random_targets_swarm_plot()

# %%
