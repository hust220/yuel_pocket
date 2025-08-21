import os
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../../')
from src.db_utils import db_connection
import pickle
from src.lightning import YuelPocket
from yuel_pocket import parse_molecule, parse_protein_structure, calculate_protein_contacts, prepare_model_input
from Bio.PDB import PDBParser
from rdkit import Chem
import io
import submitit

def create_probe_moad_table():
    """Create a table to store pocket predictions for all protein-probe pairs."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS probe_moad_predictions (
                id SERIAL PRIMARY KEY,
                pname TEXT NOT NULL,
                probe_name TEXT NOT NULL,
                pocket_pred BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pname, probe_name)
            );
            CREATE INDEX IF NOT EXISTS idx_probe_moad_pname ON probe_moad_predictions(pname);
            CREATE INDEX IF NOT EXISTS idx_probe_moad_probe ON probe_moad_predictions(probe_name);
        """)
        conn.commit()

def get_all_proteins():
    """Get all protein names from raw_datasets table."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT protein_name FROM raw_datasets")
        return [row[0] for row in cur.fetchall()]

def get_all_probes():
    """Get all probe names from probe_set table."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT lname FROM probe_set")
        return [row[0] for row in cur.fetchall()]

def get_protein_data(pname):
    """Get protein data from raw_datasets table."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT protein_pos, protein_one_hot, protein_contacts, protein_backbone
            FROM raw_datasets 
            WHERE protein_name = %s
            LIMIT 1
        """, (pname,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No data found for protein {pname}")
        
        # Unpickle the data
        protein_pos = pickle.loads(result[0])
        protein_one_hot = pickle.loads(result[1])
        protein_contacts = pickle.loads(result[2])
        protein_backbone = pickle.loads(result[3])

    return protein_pos, protein_one_hot, protein_contacts, protein_backbone

def get_ligand_data(lname):
    """Get ligand SDF content."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT mol 
            FROM ligands 
            WHERE name = %s
        """, (lname,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No SDF found for ligand {lname}")
        
        # Convert bytea to string
        sdf_content = result[0].tobytes().decode('utf-8')

        # Use RDKit's direct string reading
        ligand = Chem.MolFromMolBlock(sdf_content, sanitize=False, strictParsing=False)

        mol_pos, mol_one_hot, mol_bonds = parse_molecule(ligand)

    return mol_pos, mol_one_hot, mol_bonds

def submit_probe_jobs(num_probes=15):
    """
    Submit jobs for probe analysis using submitit.
    Each probe will be processed in a separate job.
    """
    # Create executor
    executor = submitit.AutoExecutor(folder="probe_jobs")
    
    # Get current directory and project root
    current_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # Configure the executor with your SLURM parameters
    executor.update_parameters(
        name="probe_analysis",
        slurm_account="nxd338_nih",
        slurm_partition="mgc-nih",
        nodes=1,
        slurm_ntasks_per_node=1,
        slurm_gpus_per_task=1,
        slurm_time="48:00:00",
        slurm_array_parallelism=15,  # Run at most 15 jobs in parallel
        slurm_srun_args=["--cpu-bind=none"],  # Disable CPU binding
        slurm_setup=[
            f"cd {current_dir}",  # Change to the script directory
            "module load miniconda/3",
            "conda activate torch",
            f"export PYTHONPATH={project_root}:$PYTHONPATH"  # Add project root to Python path
        ]  # Setup conda environment
    )
    
    # Create jobs for each probe
    jobs = []
    for probe_id in range(num_probes):
        # Create a job for each probe
        job = executor.submit(
            predict_pockets_for_probe,
            probe_id=probe_id
        )
        jobs.append(job)
    
    print(f"Submitted {len(jobs)} jobs")
    return jobs

def predict_pockets_for_probe(probe_id):
    """
    Predict pockets for all proteins using a specific probe.
    This function will be executed in each job.
    """
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '../../models/moad_bs8_date06-06_time11-19-53.016800/last.ckpt'
    model = YuelPocket.load_from_checkpoint(model_path, map_location=device).eval().to(device)
    
    # Get all proteins and the specific probe
    proteins = get_all_proteins()
    probes = get_all_probes()
    probe = probes[probe_id]
    
    print(f"Processing probe {probe} (ID: {probe_id})")
    print(f"Found {len(proteins)} proteins to process")
    
    # Create prediction table
    create_probe_moad_table()

    # Get probe data once
    try:
        mol_pos, mol_one_hot, mol_bonds = get_ligand_data(probe)
    except Exception as e:
        print(f"Error getting data for probe {probe}: {str(e)}")
        return
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Process each protein with this probe
        with tqdm(total=len(proteins), desc=f"Processing proteins with probe {probe}") as pbar:
            for pname in proteins:
                try:
                    # Check if prediction already exists
                    cur.execute("""
                        SELECT id FROM probe_moad_predictions 
                        WHERE pname = %s AND probe_name = %s
                    """, (pname, probe))
                    if cur.fetchone() is not None:
                        pbar.update(1)
                        continue
                    
                    # Get protein data from raw_datasets
                    print(f"Getting protein data for {pname}")
                    protein_pos, protein_one_hot, protein_contacts, protein_backbone = get_protein_data(pname)

                    data = prepare_model_input(
                        protein_pos, protein_one_hot, protein_contacts, protein_backbone,
                        mol_pos, mol_one_hot, mol_bonds, device
                    )
                    
                    # Get model prediction
                    with torch.no_grad():
                        pocket_pred = model.forward(data)
                        pocket_pred = pocket_pred.squeeze().cpu().numpy()
                    
                    # Store results
                    cur.execute("""
                        INSERT INTO probe_moad_predictions (pname, probe_name, pocket_pred)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (pname, probe_name) DO UPDATE 
                        SET pocket_pred = EXCLUDED.pocket_pred
                    """, (pname, probe, pickle.dumps(pocket_pred)))
                    conn.commit()
                    
                except Exception as e:
                    print(f"Error processing {pname}-{probe}: {str(e)}")
                    raise e
                finally:
                    pbar.update(1)
    
    print(f"Completed processing probe {probe}")

if __name__ == "__main__":
    num_probes = 15
    # For local testing, directly call predict_pockets_for_probe
    # predict_pockets_for_probe(0)  # Test with first probe only
    
    # For cluster submission, uncomment the following line:
    submit_probe_jobs(num_probes)
