from __future__ import annotations

import sys
import concurrent.futures
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import gc
import shutil

# --- 配置 ---
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

BASE_DIR = Path("./data/2024-06/v2").resolve()
# 输出目录：data/plinder/parquet/
OUTPUT_DIR = PROJ_ROOT / "data" / "plinder" / "parquet"
MAX_ITEMS: int | None = None

# 🚀 优化配置
PARQUET_BATCH_SIZE = 10000  # 每个文件包含的条目数
MAX_WORKERS = 6 
# --- /配置 ---

def _log(msg: str, pbar: tqdm | None = None) -> None:
    if pbar is not None:
        tqdm.write(msg)
    else:
        print(msg)

def load_split_map(split_parquet: Path) -> dict[str, dict]:
    cols = [
        "system_id", "uniqueness", "split", "cluster", "cluster_for_val_split", 
        "system_pass_validation_criteria", "system_pass_statistics_criteria", 
        "system_proper_num_ligand_chains", "system_proper_pocket_num_residues", 
        "system_proper_num_interactions", "system_proper_ligand_max_molecular_weight", 
        "system_has_binding_affinity", "system_has_apo_or_pred",
    ]
    table = pq.read_table(split_parquet, columns=cols)
    df = table.to_pandas()
    df.set_index("system_id", inplace=True)
    return df.to_dict('index')

def iter_system_entries(zip_path: Path):
    """🚀 优化版本：一次遍历完成所有解析，减少内存拷贝"""
    systems = defaultdict(lambda: {"receptor": None, "ligands": []})
    
    with ZipFile(zip_path) as zf:
        # 🚀 只遍历一次 namelist
        for name in zf.namelist():
            if not name or "/" not in name:
                continue
            
            # 🚀 避免创建 Path 对象
            sys_dir = name.split("/", 1)[0]
            
            if name.endswith("receptor.pdb"):
                with zf.open(name) as f:
                    systems[sys_dir]["receptor"] = f.read().decode("utf-8", "ignore")
            elif "ligand_files" in name and name.endswith(".sdf"):
                with zf.open(name) as f:
                    systems[sys_dir]["ligands"].append(f.read().decode("utf-8", "ignore"))
    
    # 🚀 使用生成器，避免一次性构建大列表
    for sys_dir, data in systems.items():
        if data["receptor"] is None and not data["ligands"]:
            continue
        
        ligand_sdf = "\n$$$$\n".join(data["ligands"]) if data["ligands"] else None
        yield (sys_dir, data["receptor"], ligand_sdf)

COLUMNS = [
    "system_id", "split", "cluster", "cluster_for_val_split", "uniqueness",
    "receptor_pdb", "ligand_sdf", 
    "system_pass_validation_criteria", "system_pass_statistics_criteria",
    "system_proper_num_ligand_chains", "system_proper_pocket_num_residues",
    "system_proper_num_interactions", "system_proper_ligand_max_molecular_weight",
    "system_has_binding_affinity", "system_has_apo_or_pred"
]

def save_batch_to_parquet(rows: list[tuple], output_dir: Path, batch_idx: int):
    """将批次数据保存为 Parquet 文件"""
    if not rows:
        return None
    
    filename = output_dir / f"part_{batch_idx:05d}.parquet"
    df = pd.DataFrame(rows, columns=COLUMNS)
    
    # 使用 PyArrow 写入
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filename, compression='snappy')
    return filename

def main():
    _log(f"💾 [info] BASE_DIR={BASE_DIR}")
    _log(f"� [info] OUTPUT_DIR={OUTPUT_DIR}")
    
    systems_dir = BASE_DIR / "systems"
    split_path = BASE_DIR / "splits" / "split.parquet"
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    _log("️ [info] loading split map...")
    split_map = load_split_map(split_path)
    _log(f"🗺️ [info] split map loaded: {len(split_map)} systems")
    
    batch = []
    processed = 0
    file_counter = 0
    
    zip_files = sorted(systems_dir.glob("*.zip"))
    _log(f"📦 [info] found {len(zip_files)} zip buckets; BATCH_SIZE={PARQUET_BATCH_SIZE}, MAX_WORKERS={MAX_WORKERS}")
    
    pbar_total = MAX_ITEMS if MAX_ITEMS is not None else len(split_map)
    pbar_zip = tqdm(total=len(zip_files), desc="Processing zip files", unit="zip")
    pbar_items = tqdm(total=pbar_total, desc="Processing systems", unit="sys")
    
    # 🛑 使用有界队列模式，防止内存爆炸
    zip_iterator = iter(zip_files)
    active_futures = set()
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    try:
        # 初始提交
        for _ in range(MAX_WORKERS * 2):
            try:
                z_file = next(zip_iterator)
                future = executor.submit(iter_system_entries, z_file)
                active_futures.add(future)
            except StopIteration:
                break
        
        while active_futures:
            # 等待至少一个任务完成
            done, active_futures = concurrent.futures.wait(
                active_futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                pbar_zip.update(1)
                try:
                    system_entries = future.result()
                    
                    # 提交新任务(如果有)
                    try:
                        next_z = next(zip_iterator)
                        new_future = executor.submit(iter_system_entries, next_z)
                        active_futures.add(new_future)
                    except StopIteration:
                        pass
                    
                    # 处理数据
                    for system_id, receptor, ligand_sdf in system_entries:
                        meta = split_map.get(system_id, {})
                        
                        row_data = (
                            system_id, meta.get("split"), meta.get("cluster"),
                            meta.get("cluster_for_val_split"), meta.get("uniqueness"),
                            receptor, ligand_sdf,
                            meta.get("system_pass_validation_criteria"), meta.get("system_pass_statistics_criteria"),
                            meta.get("system_proper_num_ligand_chains"), meta.get("system_proper_pocket_num_residues"),
                            meta.get("system_proper_num_interactions"), meta.get("system_proper_ligand_max_molecular_weight"),
                            meta.get("system_has_binding_affinity"), meta.get("system_has_apo_or_pred"),
                        )
                        batch.append(row_data)
                        processed += 1
                        pbar_items.update(1)
                        
                        # 🚀 达到批次大小，写入 Parquet
                        if len(batch) >= PARQUET_BATCH_SIZE:
                            saved_path = save_batch_to_parquet(batch, OUTPUT_DIR, file_counter)
                            _log(f"💾 [info] Saved {saved_path.name} ({len(batch)} items)", pbar_items)
                            file_counter += 1
                            batch.clear()
                            
                            # 强制 GC
                            if file_counter % 10 == 0:
                                gc.collect()
                        
                        if MAX_ITEMS is not None and processed >= MAX_ITEMS:
                            break

                    if MAX_ITEMS is not None and processed >= MAX_ITEMS:
                        break

                except Exception as exc:
                    _log(f"⚠️ [error] Exception: {exc}", pbar_items)
            
            if MAX_ITEMS is not None and processed >= MAX_ITEMS:
                executor.shutdown(wait=False, cancel_futures=True)
                break

        # 处理剩余批次
        if batch:
            saved_path = save_batch_to_parquet(batch, OUTPUT_DIR, file_counter)
            _log(f"💾 [info] Saved {saved_path.name} ({len(batch)} items)", pbar_items)
            
    finally:
        executor.shutdown(wait=True)
        pbar_zip.close()
        pbar_items.close()
        _log(f"✅ [info] Done. Processed {processed} systems. Output in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()