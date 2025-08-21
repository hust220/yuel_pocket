# %%

import sys
import pandas as pd
from pathlib import Path
sys.path.append('../..')
from src.db_utils import db_connection
from tqdm import tqdm

def add_coreset_columns():
    """Add columns for CoreSet.dat data to casf2016 table"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Add new columns if they don't exist
        cur.execute("""
            ALTER TABLE casf2016 
            ADD COLUMN IF NOT EXISTS resolution FLOAT,
            ADD COLUMN IF NOT EXISTS year INTEGER,
            ADD COLUMN IF NOT EXISTS log_ka FLOAT,
            ADD COLUMN IF NOT EXISTS ka_type VARCHAR(2),
            ADD COLUMN IF NOT EXISTS ka_value TEXT,
            ADD COLUMN IF NOT EXISTS target INTEGER
        """)
        conn.commit()

def parse_ka_string(ka_str):
    """Parse Ka string like 'Ki=1300uM' into (type, value)"""
    ka_type, value = ka_str.split('=')
    return ka_type, value

def process_coreset_data(coreset_path):
    """Process CoreSet.dat data and update casf2016 table"""
    # Read CoreSet.dat, skip comment lines
    data = []
    with open(coreset_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            data.append(line.strip().split())
    
    # Add columns if they don't exist
    add_coreset_columns()
    
    # Update database
    with db_connection() as conn:
        cur = conn.cursor()
        
        for entry in tqdm(data, desc="Updating database with CoreSet data"):
            pdb_id = entry[0]
            resolution = float(entry[1])
            year = int(entry[2])
            log_ka = float(entry[3])
            ka_type, ka_value = parse_ka_string(entry[4])
            target = int(entry[5])
            
            # Update the database
            cur.execute("""
                UPDATE casf2016 
                SET resolution = %s,
                    year = %s,
                    log_ka = %s,
                    ka_type = %s,
                    ka_value = %s,
                    target = %s
                WHERE pdb_id = %s
            """, (resolution, year, log_ka, ka_type, ka_value, target, pdb_id))
        
        conn.commit()

def create_casf_screening_table():
    """Create casf_screening table if it doesn't exist"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS casf_screening (
                id SERIAL PRIMARY KEY,
                target VARCHAR(4) NOT NULL,
                ligand VARCHAR(4) NOT NULL,
                target_rank INTEGER,
                ligand_rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(target, ligand)
            );
            CREATE INDEX IF NOT EXISTS idx_casf_screening_target ON casf_screening(target);
            CREATE INDEX IF NOT EXISTS idx_casf_screening_ligand ON casf_screening(ligand);
        """)
        conn.commit()

def process_target_info(target_info_path):
    """Process TargetInfo.dat file and update casf_screening table"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Read file and skip comments
        data = []
        with open(target_info_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                data.append(line.strip().split())
        
        # Process each line
        for entry in tqdm(data, desc="Processing TargetInfo.dat"):
            target = entry[0]
            # Skip first column (target) and process each ligand
            for i, ligand in enumerate(entry[1:], 1):
                # Skip empty entries
                if not ligand:
                    continue
                    
                # Insert or update the record
                cur.execute("""
                    INSERT INTO casf_screening (target, ligand, ligand_rank)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (target, ligand)
                    DO UPDATE SET ligand_rank = EXCLUDED.ligand_rank
                """, (target, ligand, i))
        
        conn.commit()

def process_ligand_info(ligand_info_path):
    """Process LigandInfo.dat file and update casf_screening table"""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Read file and skip comments
        data = []
        with open(ligand_info_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                data.append(line.strip().split())
        
        # Process each line
        for entry in tqdm(data, desc="Processing LigandInfo.dat"):
            ligand = entry[0]
            group = entry[1]
            # Process each target (starting from index 2)
            for i, target in enumerate(entry[2:], 1):
                # Skip empty entries
                if not target:
                    continue
                    
                # Insert or update the record
                cur.execute("""
                    INSERT INTO casf_screening (target, ligand, target_rank)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (target, ligand)
                    DO UPDATE SET target_rank = EXCLUDED.target_rank
                """, (target, ligand, i))
        
        conn.commit()

def process_screening_data():
    """Process both TargetInfo.dat and LigandInfo.dat files"""
    # Get file paths
    data_dir = Path(__file__).parent / "power_screening"
    target_info_path = data_dir / "TargetInfo.dat"
    ligand_info_path = data_dir / "LigandInfo.dat"
    
    # Create table
    create_casf_screening_table()
    
    # Process both files
    process_target_info(target_info_path)
    process_ligand_info(ligand_info_path)
    
    print("CASF screening data has been successfully processed!")

if __name__ == "__main__":
    coreset_path = Path(__file__).parent / "power_screening" / "CoreSet.dat"
    # process_coreset_data(coreset_path)
    
    process_screening_data()

# %%
