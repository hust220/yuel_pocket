#%%

from tqdm import tqdm
import sys
sys.path.append('../../')
from src.db_utils import db_connection, add_column
import pickle
from src.pdb_utils import Structure
from io import StringIO

def get_pocket(lname):
    import pickle
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT is_pocket 
            FROM processed_datasets 
            WHERE lname = %s
        """, (lname,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No data found for ligand {lname}")
        
        # Convert bytea to numpy array using pickle
        is_pocket = pickle.loads(result[0])
    
    return is_pocket

def get_pocket_pred(id):
    import pickle
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT pocket_pred 
            FROM moad_test_results 
            WHERE id = %s
        """, (id,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No data found for id {id}")
        
        # Convert bytea to numpy array using pickle
        pocket_pred = pickle.loads(result[0]).squeeze()
    
    return pocket_pred

def get_protein_positions(lname):
    """Get protein CA atom positions from database.
    
    Args:
        lname: ligand name to get protein coordinates from database
    
    Returns:
        numpy array: protein CA atom positions with shape (n_residues, 3)
    """
    import pickle
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT protein_pos 
            FROM raw_datasets 
            WHERE ligand_name = %s
        """, (lname,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No data found for ligand {lname}")
        
        # Convert bytea to numpy array using pickle
        protein_pos = pickle.loads(result[0])
    
    return protein_pos

def get_ligand_positions(lname):
    """Get ligand atom positions from database.
    
    Args:
        lname: ligand name to get ligand coordinates from database
    
    Returns:
        numpy array: ligand atom positions with shape (n_atoms, 3)
    """
    import pickle
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT molecule_pos 
            FROM raw_datasets 
            WHERE ligand_name = %s
        """, (lname,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No data found for ligand {lname}")
        
        # Convert bytea to numpy array using pickle
        ligand_pos = pickle.loads(result[0])
    
    return ligand_pos

def get_protein_pdb(lname):
    """Get protein PDB content from database.
    
    Args:
        lname: ligand name, protein name is lname[:-1]
    
    Returns:
        str: PDB content
    """
    protein_name = lname[:4]
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT pdb 
            FROM proteins 
            WHERE name = %s
        """, (protein_name,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No PDB found for protein {protein_name}")
        
        # Convert bytea to string
        pdb_content = result[0].tobytes().decode('utf-8')
    
    return pdb_content

def get_id_from_lname(lname):
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id 
            FROM processed_datasets 
            WHERE lname = %s
        """, (lname,))
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No data found for ligand {lname}")
        id = result[0]
    return id

def get_protein_allatom_positions(lname):
    """Get protein all-atom positions from PDB file.
    Only consider residues that have CA atoms.
    
    Args:
        lname: ligand name to get protein coordinates (protein name is lname[:4])
    
    Returns:
        numpy array: protein all-atom positions with shape (n_atoms, 3)
        list: residue indices for each atom position
    """
    protein_name = lname[:4]
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT pdb 
            FROM proteins 
            WHERE name = %s
        """, (protein_name,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No PDB found for protein {protein_name}")
        
        # Convert bytea to string
        pdb_content = result[0].tobytes().decode('utf-8')
    
    # Parse PDB content using our Structure class
    structure = Structure()
    structure.read(StringIO(pdb_content))
    
    # Initialize list to store positions
    all_positions = []
    
    # Process each residue
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if residue has CA atom
                has_ca = False
                for atom in residue:
                    if atom.atom_name == 'CA':
                        has_ca = True
                        break
                
                # If residue has CA atom, collect all atom positions
                if has_ca:
                    all_positions.append([atom.get_coord() for atom in residue])

    return all_positions

def store_protein_allatom_positions():
    """Store protein all-atom positions for each row in moad_test_results.
    Extract positions from PDB files, only considering residues with CA atoms.
    Store both positions and residue indices.
    Optimized version with caching and batch processing.
    """
    # Add necessary columns if they don't exist
    add_column('moad_test_results', 'protein_allatom_pos', 'BYTEA')
    
    with db_connection() as conn:
        cur = conn.cursor()
        # Get all rows from moad_test_results
        cur.execute("""
            SELECT m.id, p.lname
            FROM moad_test_results m
            JOIN processed_datasets p ON m.id = p.id
            ORDER BY p.pname  -- Order by protein name for better caching
        """)
        rows = cur.fetchall()
        
        print(f"Processing {len(rows)} rows")
        
        # Cache for protein all-atom positions to avoid redundant PDB parsing
        protein_cache = {}
        batch_updates = []
        batch_size = 100  # Process in batches
        
        iprocessed = 0
        
        # Process each row
        for id, lname in tqdm(rows, total=len(rows), desc="Storing protein all-atom positions"):
            try:
                protein_name = lname[:4]
                
                # Check if we already have this protein's positions in cache
                if protein_name not in protein_cache:
                    # Get all-atom positions and cache them
                    allatom_pos = get_protein_allatom_positions(lname)
                    protein_cache[protein_name] = allatom_pos
                else:
                    # Use cached positions
                    allatom_pos = protein_cache[protein_name]
                
                # Add to batch
                batch_updates.append((pickle.dumps(allatom_pos), id))
                
                # Execute batch when it reaches batch_size
                if len(batch_updates) >= batch_size:
                    cur.executemany("""
                        UPDATE moad_test_results 
                        SET protein_allatom_pos = %s
                        WHERE id = %s
                    """, batch_updates)
                    conn.commit()
                    iprocessed += len(batch_updates)
                    batch_updates = []
                
            except Exception as e:
                print(f"Error processing {lname} (id: {id}): {str(e)}")
                continue
        
        # Execute remaining batch
        if batch_updates:
            cur.executemany("""
                UPDATE moad_test_results 
                SET protein_allatom_pos = %s
                WHERE id = %s
            """, batch_updates)
            conn.commit()
            iprocessed += len(batch_updates)
        
        print(f"Successfully processed {iprocessed} rows")
        print(f"Used cache for {len(protein_cache)} unique proteins")

def calculate_num_ligands():
    """Calculate number of ligands for each protein in moad_test_results.
    Optimized version using a single SQL UPDATE statement.
    """
    # Add necessary column if it doesn't exist
    add_column('moad_test_results', 'num_ligands', 'INTEGER')
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        print("Calculating number of ligands...")
        
        # Update all rows with a single SQL statement using a subquery
        cur.execute("""
            UPDATE moad_test_results 
            SET num_ligands = ligand_counts.ligand_count
            FROM (
                SELECT protein_name, COUNT(*) as ligand_count
                FROM raw_datasets
                GROUP BY protein_name
            ) AS ligand_counts,
            processed_datasets p 
            WHERE moad_test_results.id = p.id
            AND ligand_counts.protein_name = p.pname
        """)
        
        updated_rows = cur.rowcount
        conn.commit()
        
        print(f"Successfully updated {updated_rows} rows")
        
        # Print some statistics
        cur.execute("SELECT MIN(num_ligands), MAX(num_ligands), AVG(num_ligands)::numeric(10,2) FROM moad_test_results WHERE num_ligands IS NOT NULL")
        result = cur.fetchone()
        if result and result[0] is not None:
            min_val, max_val, avg_val = result
            print(f"\nStatistics:")
            print(f"Minimum number of ligands: {min_val}")
            print(f"Maximum number of ligands: {max_val}")
            print(f"Average number of ligands: {avg_val}")
            
            # Print distribution
            cur.execute("""
                SELECT num_ligands, COUNT(*) as count
                FROM moad_test_results
                WHERE num_ligands IS NOT NULL
                GROUP BY num_ligands
                ORDER BY num_ligands
            """)
            print("\nDistribution:")
            for num, count in cur.fetchall():
                print(f"{num} ligands: {count} proteins")
        else:
            print("No data was updated - check if the tables contain matching records.")

def calculate_pocket_clusters():
    query = """
        SELECT m.id, p.lname
        FROM moad_test_results m
        JOIN processed_datasets p ON m.id = p.id
    """
    parallel_process_clusters(query, process_pocket_clusters_chunk, 'clusters')

def calculate_true_pocket_clusters():
    query = """
        SELECT m.id, p.lname
        FROM moad_test_results m
        JOIN processed_datasets p ON m.id = p.id
    """
    parallel_process_clusters(query, process_true_pocket_clusters_chunk, 'true_clusters')

def calculate_all_pockets_clusters():
    query = """
        SELECT DISTINCT p.name, p.all_pockets
        FROM proteins p
        JOIN processed_datasets pd ON p.name = pd.pname
        JOIN moad_test_results m ON pd.id = m.id
        WHERE p.all_pockets IS NOT NULL
    """
    parallel_process_proteins(query, process_all_pockets_clusters_chunk, 'proteins', 'all_clusters')

def calculate_pocket_centers():
    """Calculate pocket centers for all entries in moad_test_results table using FFT-based approach."""
    
    # Add necessary columns if they don't exist
    add_column('moad_test_results', 'pocket_centers', 'BYTEA')
    add_column('moad_test_results', 'center_scores', 'BYTEA')
    
    query = """
        SELECT m.id, p.lname
        FROM moad_test_results m
        JOIN processed_datasets p ON m.id = p.id
    """
    parallel_process_centers(query)

def process_with_progress(args):
    chunk_data, process_func = args
    return process_func(chunk_data)

def parallel_process_clusters(query, process_func, column_name, num_cores=8):
    import multiprocessing as mp
    from multiprocessing import Pool
    import math
    
    add_column('moad_test_results', column_name, 'BYTEA')
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        
        print(f"Processing {len(rows)} rows using {num_cores} CPU cores")
        
        # chunk_size = math.ceil(len(rows) / num_cores)
        chunk_size = 10
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        
        # Prepare arguments for each chunk
        chunk_args = [(chunk, process_func) for chunk in chunks]
        
        with Pool(processes=num_cores) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_with_progress, chunk_args),
                total=len(chunks),
                desc="Processing chunks"
            ))
        
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        if all_results:
            batch_size = 100
            for i in tqdm(range(0, len(all_results), batch_size), desc="Updating database"):
                batch = all_results[i:i + batch_size]
                cur.executemany(f"""
                    UPDATE moad_test_results 
                    SET {column_name} = %s 
                    WHERE id = %s
                """, [(data, id) for id, data in batch])
                conn.commit()
        
        print(f"Successfully processed {len(all_results)} rows")

def parallel_process_proteins(query, process_func, table_name, column_name, num_cores=8):
    import multiprocessing as mp
    from multiprocessing import Pool
    import math
    
    add_column(table_name, column_name, 'BYTEA')
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        
        processed_rows = []
        for row in rows:
            if len(row) == 2 and isinstance(row[1], memoryview):
                processed_rows.append((row[0], row[1].tobytes()))
            else:
                processed_rows.append(row)
        
        print(f"Processing {len(processed_rows)} proteins using {num_cores} CPU cores")
        
        # chunk_size = math.ceil(len(rows) / num_cores)
        chunk_size = 10
        chunks = [processed_rows[i:i + chunk_size] for i in range(0, len(processed_rows), chunk_size)]
        
        # Prepare arguments for each chunk
        chunk_args = [(chunk, process_func) for chunk in chunks]
        
        with Pool(processes=num_cores) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_with_progress, chunk_args),
                total=len(chunks),
                desc="Processing chunks"
            ))
        
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        if all_results:
            batch_size = 50
            for i in tqdm(range(0, len(all_results), batch_size), desc="Updating database"):
                batch = all_results[i:i + batch_size]
                cur.executemany(f"""
                    UPDATE {table_name} 
                    SET {column_name} = %s 
                    WHERE name = %s
                """, [(data, name) for name, data in batch])
                conn.commit()
        
        print(f"Successfully processed {len(all_results)} proteins")

def process_pocket_clusters_chunk(chunk_data):
    import pickle
    from io import StringIO
    from src.clustering import cluster_pocket_predictions
    
    results = []
    with db_connection() as conn:
        cur = conn.cursor()
        
        for id, lname in chunk_data:
            try:
                cur.execute("SELECT pocket_pred FROM moad_test_results WHERE id = %s", (id,))
                pocket_pred = pickle.loads(cur.fetchone()[0]).squeeze()
                
                pdb_content = get_protein_pdb(lname)
                structure = Structure()
                structure.read(StringIO(pdb_content))
                
                clusters = cluster_pocket_predictions(pocket_pred, structure, cutoff=0.06, distance_threshold=5.0)
                results.append((id, pickle.dumps(clusters)))
                
            except Exception as e:
                print(f"Error processing {lname} (id: {id}): {str(e)}")
                continue
    
    return results

def process_true_pocket_clusters_chunk(chunk_data):
    import pickle
    import numpy as np
    from io import StringIO
    from src.clustering import cluster_pocket_predictions
    
    results = []
    with db_connection() as conn:
        cur = conn.cursor()
        
        for id, lname in chunk_data:
            try:
                cur.execute("SELECT is_pocket FROM processed_datasets WHERE lname = %s", (lname,))
                is_pocket = pickle.loads(cur.fetchone()[0])
                
                pdb_content = get_protein_pdb(lname)
                structure = Structure()
                structure.read(StringIO(pdb_content))
                
                pocket_prob = np.where(is_pocket, 1.0, 0.0)
                clusters = cluster_pocket_predictions(pocket_prob, structure, cutoff=0.5, distance_threshold=5.0)
                results.append((id, pickle.dumps(clusters)))
                
            except Exception as e:
                print(f"Error processing {lname} (id: {id}): {str(e)}")
                raise e
    
    return results

def process_all_pockets_clusters_chunk(chunk_data):
    import pickle
    from io import StringIO
    from src.clustering import cluster_pocket_predictions
    
    results = []
    with db_connection() as conn:
        cur = conn.cursor()
        
        for pname, all_pockets in chunk_data:
            try:
                # Convert memoryview to bytes before unpickling
                if isinstance(all_pockets, memoryview):
                    all_pockets = all_pockets.tobytes()
                all_pockets = pickle.loads(all_pockets)
                
                cur.execute("SELECT pdb FROM proteins WHERE name = %s", (pname,))
                pdb_content = cur.fetchone()[0].tobytes().decode('utf-8')
                structure = Structure()
                structure.read(StringIO(pdb_content))
                
                all_pockets = all_pockets.astype(int)
                clusters = cluster_pocket_predictions(all_pockets, structure, cutoff=0.5, distance_threshold=5.0)
                results.append((pname, pickle.dumps(clusters)))
                
            except Exception as e:
                print(f"Error processing protein {pname}: {str(e)}")
                raise e
    
    return results

def parallel_process_centers(query, num_cores=8):
    """Parallel processing specifically for pocket centers and scores."""
    import multiprocessing as mp
    from multiprocessing import Pool
    import math
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        
        print(f"Processing {len(rows)} rows using {num_cores} CPU cores")
        
        chunk_size = 1
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        
        # Prepare arguments for each chunk
        chunk_args = [(chunk, process_pocket_centers_chunk) for chunk in chunks]
        
        with Pool(processes=num_cores) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_with_progress, chunk_args),
                total=len(chunks),
                desc="Processing chunks"
            ))
        
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        if all_results:
            batch_size = 100
            for i in tqdm(range(0, len(all_results), batch_size), desc="Updating database"):
                batch = all_results[i:i + batch_size]
                cur.executemany("""
                    UPDATE moad_test_results 
                    SET pocket_centers = %s, center_scores = %s
                    WHERE id = %s
                """, [(centers, scores, id) for id, centers, scores in batch])
                conn.commit()
        
        print(f"Successfully processed {len(all_results)} rows")

def process_pocket_centers_chunk(chunk_data):
    """Process a chunk of data to calculate pocket centers.
    
    Args:
        chunk_data: List of (id, lname) tuples
        
    Returns:
        list: List of (id, pickled_centers, pickled_scores) tuples
    """
    import pickle
    from io import StringIO
    from src.pocket_utils import find_pocket_centers
    import numpy as np
    
    results = []
    with db_connection() as conn:
        cur = conn.cursor()
        
        for id, lname in chunk_data:
            try:
                # Get pocket predictions
                cur.execute("SELECT pocket_pred FROM moad_test_results WHERE id = %s", (id,))
                pocket_pred = pickle.loads(cur.fetchone()[0]).squeeze()
                
                # Get protein structure
                pdb_content = get_protein_pdb(lname)
                structure = Structure()
                structure.read(StringIO(pdb_content))
                
                # Calculate pocket centers
                centers_data = find_pocket_centers(structure, pocket_pred)
                if centers_data is not None:
                    # Only store centers and their scores
                    centers = centers_data['centers']  # shape: (n_clusters, 3)
                    scores = centers_data['center_scores']  # shape: (n_clusters,)
                    
                    # Convert to numpy arrays if they aren't already
                    centers = np.array(centers)
                    scores = np.array(scores)
                    
                    results.append((id, pickle.dumps(centers), pickle.dumps(scores)))
                
            except Exception as e:
                print(f"Error processing {lname} (id: {id}): {str(e)}")
                continue
    
    return results

if __name__ == "__main__":    
    # store_protein_allatom_positions()
    # calculate_num_ligands()
    # calculate_pocket_clusters()
    # calculate_true_pocket_clusters()
    # calculate_all_pockets_clusters()
    calculate_pocket_centers()


# %%
