#%%

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.db_utils import db_connection, add_column
from src.pdb_utils import download_pdb, Structure
import time
from typing import Optional
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('import_asd.log'),
        logging.StreamHandler()
    ]
)

def create_allobench_table():
    """创建allobench表"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # 创建表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS allobench (
                id SERIAL PRIMARY KEY,
                protein_asd_id VARCHAR(255),
                gene VARCHAR(255),
                organism VARCHAR(255),
                uniprot_id VARCHAR(255),
                pdb_id VARCHAR(255),
                protein_class VARCHAR(255),
                ec_number TEXT[],
                resolution FLOAT,
                experimental_method VARCHAR(255),
                oligomeric_state VARCHAR(255),
                stoichiometry TEXT[],
                allosteric_site_residues TEXT[],
                active_site_residues TEXT[],
                pdb_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pdb_id ON allobench(pdb_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_uniprot_id ON allobench(uniprot_id)")
        
        conn.commit()

def process_pdb(pdb_id: str) -> Optional[str]:
    """下载并处理PDB文件"""
    try:
        # 下载PDB文件
        structure = Structure()
        structure.read(download_pdb(pdb_id))
        return structure.to_pdb()
    except Exception as e:
        logging.error(f"Error processing PDB {pdb_id}: {str(e)}")
        return None

def import_data():
    """导入数据到数据库"""
    # 读取CSV文件
    df = pd.read_csv('ASD_High_Resolution.csv')
    
    # 创建表
    create_allobench_table()
    
    # 连接数据库
    with db_connection() as conn:
        cur = conn.cursor()
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=len(df), desc="Importing entries", unit="entries")
        
        # 遍历每一行
        for idx, row in df.iterrows():
            try:
                # 处理列表类型的字段
                ec_number = row['EC Number'].strip('[]').split(',') if pd.notna(row['EC Number']) else []
                stoichiometry = row['Stoichiometry'].strip('[]').split(',') if pd.notna(row['Stoichiometry']) else []
                allosteric_site_residues = row['Allosteric Site Residues'].strip('[]').split(',') if pd.notna(row['Allosteric Site Residues']) else []
                active_site_residues = row['Active Site Residues'].strip('[]').split(',') if pd.notna(row['Active Site Residues']) else []
                
                # 清理列表中的字符串
                ec_number = [x.strip().strip("'") for x in ec_number]
                stoichiometry = [x.strip().strip("'") for x in stoichiometry]
                allosteric_site_residues = [x.strip().strip("'") for x in allosteric_site_residues]
                active_site_residues = [x.strip().strip("'") for x in active_site_residues]
                
                # 更新进度条描述
                pbar.set_description(f"Processing PDB {row['PDB ID']}")
                
                # 下载并获取PDB结构
                pdb_content = process_pdb(row['PDB ID'])
                
                # 插入数据
                cur.execute("""
                    INSERT INTO allobench (
                        protein_asd_id, gene, organism, uniprot_id, pdb_id,
                        protein_class, ec_number, resolution, experimental_method,
                        oligomeric_state, stoichiometry, allosteric_site_residues,
                        active_site_residues, pdb_content
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    row['Protein ASD ID'], row['Gene'], row['Organism'], row['UniProt ID'],
                    row['PDB ID'], row['Protein Class'], ec_number, row['Resolution'],
                    row['Experimental Method'], row['Oligomeric State'], stoichiometry,
                    allosteric_site_residues, active_site_residues, pdb_content
                ))
                
                conn.commit()
                logging.info(f"Successfully imported data for PDB ID: {row['PDB ID']}")
                
            except Exception as e:
                logging.error(f"Error importing row {idx} (PDB ID: {row['PDB ID']}): {str(e)}")
                conn.rollback()
                continue
            finally:
                # 更新进度条
                pbar.update(1)
            
            # 每处理50条记录暂停一下，避免过度请求PDB服务器
            if idx % 50 == 0:
                time.sleep(5)
        
        # 关闭进度条
        pbar.close()

def table_add_missing_columns():
    """Add missing columns from ASD CSV to the existing allobench table"""
    # Read CSV file
    df = pd.read_csv('ASD_High_Resolution.csv')
    
    # List of tuples containing (column_name, column_type, csv_column_name)
    new_columns = [
        ("modulator_asd_id", "TEXT", "Modulator ASD ID"),
        ("modulator_alias", "TEXT", "Modulator Alias"),
        ("modulator_chain", "TEXT", "Modulator Chain"),
        ("modulator_class", "TEXT", "Modulator Class"),
        ("allosteric_activity", "TEXT", "Allosteric Activity"),
        ("modulator_name", "TEXT", "Modulator Name"),
        ("modulator_residue_id", "TEXT", "Modulator Residue ID"),
        ("asd_function", "TEXT", "ASD Function"),
        ("position", "TEXT", "Position"),
        ("pubmed", "TEXT", "PubMed"),
        ("reference_title", "TEXT", "Reference Title"),
        ("site_overlap", "TEXT", "Site Overlap"),
        ("asd_allosteric_site_residues", "TEXT[]", "ASD Allosteric Site Residues"),
        ("map_pdb_chain_to_uniprot", "TEXT", "Map PDB Chain to UniProt"),
        ("reviewed", "BOOLEAN", "Reviewed"),
        ("protein_name", "TEXT", "Protein Name"),
        ("uniprot_in_pdb", "TEXT[]", "_uniprot_in_pdb")
    ]
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Process each column
        for db_col, col_type, csv_col in new_columns:
            try:
                # Add column
                logging.info(f"Adding column {db_col}")
                add_column("allobench", db_col, col_type)
                
                # Prepare and import data
                logging.info(f"Importing data for column {db_col}")
                
                if col_type == "TEXT[]":
                    # Handle array type columns
                    values = []
                    for _, row in df.iterrows():
                        if pd.notna(row[csv_col]):
                            arr = row[csv_col].strip('[]').split(',')
                            arr = [x.strip().strip("'") for x in arr]
                            values.append((row['PDB ID'], arr if arr else None))
                        else:
                            values.append((row['PDB ID'], None))
                
                elif col_type == "BOOLEAN":
                    # Handle boolean type
                    values = [(row['PDB ID'], str(row[csv_col]).lower() == 'true') 
                            for _, row in df.iterrows()]
                
                else:
                    # Handle text type
                    values = [(row['PDB ID'], row[csv_col] if pd.notna(row[csv_col]) else None) 
                            for _, row in df.iterrows()]
                
                # Convert values to SQL format
                values_str = ','.join(
                    cur.mogrify("(%s,%s)", x).decode('utf-8')
                    for x in values
                )
                
                # Update the column
                cur.execute(f"""
                    UPDATE allobench AS a
                    SET {db_col} = t.val
                    FROM (
                        VALUES {values_str}
                    ) AS t(pdb_id, val)
                    WHERE a.pdb_id = t.pdb_id
                """)
                
                conn.commit()
                logging.info(f"Successfully updated column {db_col}")
                
            except Exception as e:
                logging.error(f"Error processing column {db_col}: {str(e)}")
                conn.rollback()
                continue
        
        # Create index for modulator_asd_id if it doesn't exist
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_modulator_asd_id ON allobench(modulator_asd_id)")
            conn.commit()
            logging.info("Created index on modulator_asd_id")
        except Exception as e:
            logging.info(f"Index on modulator_asd_id might already exist: {str(e)}")

if __name__ == "__main__":
    # import_data()
    table_add_missing_columns()
# %%
