import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
import time
import os
from pathlib import Path

def load_db_config():
    """Load database configuration from config file"""
    config_path = Path(__file__).parent / 'db_config'
    if not config_path.exists():
        raise FileNotFoundError(f"Database configuration file not found at {config_path}")
    
    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    
    return {
        'dbname': config['DB_NAME'],
        'user': config['DB_USER'],
        'host': config['DB_HOST'],
        'port': config['DB_PORT']
    }

@contextmanager
def db_connection(dbname=None):
    """Database connection with retry mechanism"""
    attempts = 0
    conn = None
    try:
        while True:
            try:
                db_params = load_db_config()
                if dbname:
                    db_params['dbname'] = dbname
                conn = psycopg2.connect(**db_params)
                conn.autocommit = False
                break
            except psycopg2.OperationalError as e:
                if attempts < 10:
                    time.sleep(2 ** attempts)
                    attempts += 1
                    continue
                else:
                    raise
        yield conn
    finally:
        if conn:
            conn.close()

def add_column(table_name, column_name, column_type):
    """Add a column to a table if it doesn't exist.
    
    Args:
        table_name (str): Name of the table
        column_name (str): Name of the column to add
        column_type (str): PostgreSQL data type of the column
    """
    with db_connection() as conn:
        cur = conn.cursor()
        # Check if column exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND column_name = %s
            );
        """, (table_name, column_name))
        column_exists = cur.fetchone()[0]
        
        if not column_exists:
            cur.execute(f"""
                ALTER TABLE {table_name} 
                ADD COLUMN {column_name} {column_type};
            """)
            conn.commit()
            print(f"Added column {column_name} to table {table_name}")
        else:
            print(f"Column {column_name} already exists in table {table_name}") 

def db_select(table_name, column_name, condition, is_pickle=False, is_bytea=False):
    import pickle
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT {column_name} 
            FROM {table_name} 
            WHERE {condition}
        """)
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No data found for {condition}")
        
        if is_pickle:
            # Convert bytea to numpy array using pickle
            data = pickle.loads(result[0])
        elif is_bytea:
            data = result[0].tobytes().decode('utf-8')
        else:
            data = result[0]
    
    return data

