import sys
import pickle
import numpy as np
from tqdm import tqdm
import io
sys.path.append('../..')
from src.db_utils import db_connection
from yuel_pocket import parse_protein_structure
from sklearn.cluster import DBSCAN
from src.pdb_utils import Structure

# --- 新增：先存储蛋白结构解析结果到casf2016 ---
def add_protein_structure_to_casf2016():
    columns = {
        'protein_pos': 'BYTEA',
        'protein_one_hot': 'BYTEA',
        'residue_ids': 'BYTEA',
        'atom_coords': 'BYTEA',
        'protein_backbone': 'BYTEA'
    }
    with db_connection() as conn:
        cur = conn.cursor()
        # 检查并添加新列
        for col, typ in columns.items():
            cur.execute(f"""
                ALTER TABLE casf2016
                ADD COLUMN IF NOT EXISTS {col} {typ}
            """)
        conn.commit()
        # 获取所有target
        cur.execute("SELECT pdb_id FROM casf2016")
        pdb_ids = [row[0] for row in cur.fetchall()]
        print(f"Total targets to process: {len(pdb_ids)}")
        for count, pdb_id in enumerate(pdb_ids, 1):
            try:
                cur.execute("SELECT protein_pdb FROM casf2016 WHERE pdb_id = %s", (pdb_id,))
                protein_pdb = cur.fetchone()[0]
                protein_pdb = protein_pdb.tobytes().decode('utf-8')
                struct = Structure(io.StringIO(protein_pdb), skip_hetatm=False, skip_water=True)
                protein_pos, protein_one_hot, residue_ids, atom_coords, protein_backbone = parse_protein_structure(struct)
                cur.execute("""
                    UPDATE casf2016 SET protein_pos = %s, protein_one_hot = %s, residue_ids = %s, atom_coords = %s, protein_backbone = %s
                    WHERE pdb_id = %s
                """, (
                    pickle.dumps(protein_pos),
                    pickle.dumps(protein_one_hot),
                    pickle.dumps(residue_ids),
                    pickle.dumps(atom_coords),
                    pickle.dumps(protein_backbone),
                    pdb_id
                ))
                conn.commit()
                if count % 10 == 0:
                    print(f"Processed {count} targets...")
            except Exception as e:
                raise e
        print(f"\nFinished storing protein structure info in casf2016. Total processed: {count}")

# --- 聚类函数，复用analyze_pdbbind.py ---
def cluster_pocket_predictions(pocket_pred, protein_pos, cutoff=0.05, distance_threshold=5.0):
    """Cluster pocket predictions based on spatial positions of residues."""
    if len(pocket_pred) > len(protein_pos):
        pocket_pred = pocket_pred[:len(protein_pos)]
    mask = (pocket_pred > cutoff)
    selected_pos = protein_pos[mask]
    if len(selected_pos) == 0:
        print(f"Warning: No residues selected for clustering with cutoff {cutoff}")
        return np.full_like(pocket_pred, -1, dtype=int)
    clustering = DBSCAN(eps=distance_threshold, min_samples=3).fit(selected_pos)
    clusters = np.full_like(pocket_pred, -1, dtype=int)
    clusters[mask] = clustering.labels_
    return clusters

# --- 主流程 ---
def calculate_casf_pocket_clusters():
    with db_connection() as conn:
        cur = conn.cursor()
        # 检查并添加clusters列
        cur.execute("""
            ALTER TABLE casf_predictions
            ADD COLUMN IF NOT EXISTS clusters BYTEA
        """)
        conn.commit()
        # 获取所有target/ligand对
        cur.execute("""
            SELECT target, ligand, pocket_pred FROM casf_predictions
            WHERE clusters IS NULL OR clusters = ''
        """)
        entries = cur.fetchall()
        print(f"Total entries to process: {len(entries)}")
        for target, ligand, pocket_pred in tqdm(entries, desc="Clustering pockets"):
            try:
                # 从casf2016读取protein_pos
                cur.execute("""
                    SELECT protein_pos FROM casf2016 WHERE pdb_id = %s
                """, (target,))
                protein_pos = cur.fetchone()[0]
                pocket_pred = pickle.loads(pocket_pred).squeeze()
                protein_pos = pickle.loads(protein_pos)
                clusters = cluster_pocket_predictions(pocket_pred, protein_pos, cutoff=0.05, distance_threshold=5.0)
                cur.execute("""
                    UPDATE casf_predictions SET clusters = %s WHERE target = %s AND ligand = %s
                """, (pickle.dumps(clusters), target, ligand))
                conn.commit()
            except Exception as e:
                print(f"Error processing {target}-{ligand}: {str(e)}")
                raise e
                # continue
    print("\nFinished processing all entries.")

if __name__ == "__main__":
    # 先存蛋白结构到casf2016
    add_protein_structure_to_casf2016()
    # 再聚类
    calculate_casf_pocket_clusters()
