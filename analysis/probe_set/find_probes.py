#%%

import os
import torch
import numpy as np
from tqdm import tqdm
import sys, os
sys.path.append(os.path.join(__file__, '../../..'))
from src.db_utils import db_connection
import pickle
from collections import defaultdict
import random
from src.lightning import YuelPocket
from yuel_pocket import _predict_pocket
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from IPython.display import SVG, display

def create_probe_predictions_table():
    """Create a table to store pocket predictions for probe ligands."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS probe_predictions (
                id SERIAL PRIMARY KEY,
                pname TEXT NOT NULL,
                lname TEXT NOT NULL,
                pocket_pred BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pname, lname)
            );
            CREATE INDEX IF NOT EXISTS idx_probe_pred_pname ON probe_predictions(pname);
            CREATE INDEX IF NOT EXISTS idx_probe_pred_lname ON probe_predictions(lname);
        """)
        conn.commit()

def get_protein_ligand_mapping():
    """Get mapping of protein names to their ligands.
    
    Returns:
        dict: {protein_name: [ligand_names]}
    """
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT pname, lname
            FROM processed_datasets
            WHERE split = 'train'
        """)
        rows = cur.fetchall()
    
    protein_to_ligands = defaultdict(list)
    for pname, lname in rows:
        protein_to_ligands[pname].append(lname)
    
    return protein_to_ligands

def get_pocket_residues(protein_names, protein_to_ligands):
    """Get pocket residues for given proteins.
    
    Args:
        protein_names: list of protein names
        protein_to_ligands: dict mapping protein names to their ligands
    
    Returns:
        dict: {protein_name: set of pocket residue indices}
    """
    pocket_residues = {}
    
    with db_connection() as conn:
        cur = conn.cursor()
        for pname in protein_names:
            # Get ligands for this protein from the mapping
            ligands = protein_to_ligands[pname]
            
            # Get pocket residues for each ligand
            pocket_indices = set()
            for lname in ligands:
                cur.execute("""
                    SELECT is_pocket
                    FROM processed_datasets
                    WHERE lname = %s
                """, (lname,))
                is_pocket = pickle.loads(cur.fetchone()[0])
                pocket_indices.update(np.where(is_pocket == 1)[0])
            
            pocket_residues[pname] = pocket_indices
    
    return pocket_residues

def get_protein_pdb(pname):
    """Get protein PDB content.
    
    Args:
        pname: protein name
    
    Returns:
        str: PDB content
    """
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT pdb 
            FROM proteins 
            WHERE name = %s
        """, (pname,))
        
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No PDB found for protein {pname}")
        
        # Convert bytea to string
        pdb_content = result[0].tobytes().decode('utf-8')
    
    return pdb_content

def get_ligand_sdf(lname):
    """Get ligand SDF content.
    
    Args:
        lname: ligand name
    
    Returns:
        str: SDF content
    """
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
    
    return sdf_content

def get_pocket_prediction(ligand_name, batch_proteins, model, device):
    """Get pocket prediction for a ligand with all proteins in the batch.
    First try to get from database, if not found, calculate and store.
    
    Args:
        ligand_name: ligand name
        batch_proteins: list of protein names in current batch
        model: YuelPocket model
        device: torch device
    
    Returns:
        dict: {protein_name: predicted pocket residues}
    """
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get ligand SDF content (only need to get once)
        ligand_sdf = get_ligand_sdf(ligand_name)
        
        pred_pockets = {}
        for pname in batch_proteins:
            # First try to get from database
            cur.execute("""
                SELECT pocket_pred
                FROM probe_predictions
                WHERE pname = %s AND lname = %s
            """, (pname, ligand_name))
            result = cur.fetchone()
            
            if result is not None:
                # If found in database, use it
                pocket_pred = pickle.loads(result[0])
                pred_pocket_indices = set(np.where(pocket_pred > 0.1)[0])
                pred_pockets[pname] = pred_pocket_indices
                continue
            
            # If not found, calculate prediction
            try:
                # Get PDB content
                protein_pdb = get_protein_pdb(pname)
                
                # Run prediction
                pocket_pred, _ = _predict_pocket(
                    protein_pdb,
                    ligand_sdf,
                    model,
                    distance_cutoff=10.0,
                    device=device
                )
                
                # Convert predictions to numpy array
                pocket_pred = pocket_pred.cpu().numpy().squeeze()
                
                # Store prediction in database
                cur.execute("""
                    INSERT INTO probe_predictions (pname, lname, pocket_pred)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (pname, lname) DO UPDATE 
                    SET pocket_pred = EXCLUDED.pocket_pred
                """, (pname, ligand_name, pickle.dumps(pocket_pred)))
                conn.commit()
                
                # Get pocket residues (probability > 0.1)
                pred_pocket_indices = set(np.where(pocket_pred > 0.1)[0])
                pred_pockets[pname] = pred_pocket_indices
                
            except Exception as e:
                print(f"Error predicting pocket for {pname}-{ligand_name}: {str(e)}")
                continue
        
        return pred_pockets

def calculate_new_residues(pred_pockets, current_predictions, true_pockets):
    """Calculate how many new pocket residues are found.
    
    Args:
        pred_pockets: dict of {protein_name: set of predicted pocket indices}
        current_predictions: dict of {protein_name: set of current predicted pocket indices}
        true_pockets: dict of {protein_name: set of true pocket indices}
    
    Returns:
        int: number of new pocket residues found
    """
    new_residues = 0
    
    for pname in true_pockets:
        if pname not in pred_pockets:
            continue
            
        # Get true pocket residues for this protein
        true_indices = true_pockets[pname]
        
        # Get current predicted residues
        current_indices = current_predictions.get(pname, set())
        
        # Get new predicted residues
        new_indices = pred_pockets[pname]
        
        # Calculate newly found residues (intersection with true pockets that weren't found before)
        newly_found = (new_indices & true_indices) - current_indices
        new_residues += len(newly_found)
    
    return new_residues

def create_probe_set_table():
    """Create a table to store the final probe set."""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS probe_set (
                id SERIAL PRIMARY KEY,
                lname TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

def save_probe_set(probe_set):
    """Save the final probe set to the database.
    
    Args:
        probe_set: set of probe ligands
    """
    create_probe_set_table()
    
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Clear existing entries
        cursor.execute("DELETE FROM probe_set")
        
        # Insert new entries
        for ligand in probe_set:
            cursor.execute(
                "INSERT INTO probe_set (lname) VALUES (%s)",
                (ligand,)
            )
        
        conn.commit()
    
    print(f"\nSaved {len(probe_set)} ligands to probe_set table")

def find_probe_ligands(batch_size=50, max_batches=10):
    """Find probe ligands using the new algorithm.
    
    Args:
        batch_size: number of proteins in each batch
        max_batches: maximum number of batches to process
    
    Returns:
        set: set of probe ligands
        dict: coverage history
    """
    # Create predictions table
    create_probe_predictions_table()
    
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '../../models/moad_bs8_date06-06_time11-19-53.016800/last.ckpt'
    model = YuelPocket.load_from_checkpoint(model_path, map_location=device).eval().to(device)
    
    # Get all proteins
    protein_to_ligands = get_protein_ligand_mapping()
    all_proteins = list(protein_to_ligands.keys())
    random.shuffle(all_proteins)
    
    probe_set = set()
    coverage_history = []
    current_batch = 0
    consecutive_skips = 0
    
    while current_batch < max_batches and len(all_proteins) > 0:
        print(f"\nProcessing batch {current_batch + 1}")
        
        # Get next batch of proteins
        batch_proteins = all_proteins[:batch_size]
        all_proteins = all_proteins[batch_size:]
        
        # Get pocket residues for this batch
        print("Getting pocket residues...")
        batch_pocket_set = get_pocket_residues(batch_proteins, protein_to_ligands)
        
        # Initialize current predictions with probe set predictions for new proteins
        current_predictions = {}
        if probe_set:
            print("Initializing predictions with current probe set...")
            for ligand in tqdm(probe_set, desc="Initializing predictions"):
                pred_pockets = get_pocket_prediction(ligand, batch_proteins, model, device)
                if pred_pockets:
                    for pname, indices in pred_pockets.items():
                        if pname in current_predictions:
                            current_predictions[pname].update(indices)
                        else:
                            current_predictions[pname] = indices
        
        # Get all ligands for this batch from the mapping
        batch_ligands = set()
        for pname in batch_proteins:
            batch_ligands.update(protein_to_ligands[pname])
        batch_ligands = list(batch_ligands)
        random.shuffle(batch_ligands)
        
        print(f"Found {len(batch_ligands)} ligands for this batch")
        print(f"Current batch pocket set size: {sum(len(indices) for indices in batch_pocket_set.values())} residues")
        
        # Initialize coverage tracking for this batch
        total_residues = sum(len(indices) for indices in batch_pocket_set.values())
        covered_residues = sum(len(indices) for indices in current_predictions.values())
        
        # Check if pocket set is already fully covered
        if covered_residues / total_residues >= 0.8:
            print(f"Pocket set already fully covered by current probe set, skipping this batch")
            coverage_history.append({
                'batch': current_batch + 1,
                'coverage': 1.0,
                'n_ligands': len(probe_set),
                'total_residues': total_residues,
                'covered_residues': total_residues,
                'new_proteins': len(batch_proteins),
                'total_proteins': len(current_predictions),
                'skipped': True
            })
            consecutive_skips += 1
            if consecutive_skips >= 3:
                print("Consecutive skips reached 3, terminating search")
                break
            current_batch += 1
            continue
        
        # Reset consecutive skips counter if batch is not skipped
        consecutive_skips = 0
        
        # Try adding ligands one by one
        for ligand in tqdm(batch_ligands, desc="Finding probe ligands"):
            # Get pocket prediction for this ligand with all proteins in batch
            pred_pockets = get_pocket_prediction(ligand, batch_proteins, model, device)
            if not pred_pockets:  # Skip if no predictions were made
                continue
            
            # Calculate how many new residues this ligand finds
            new_residues = calculate_new_residues(pred_pockets, current_predictions, batch_pocket_set)
            
            # If this ligand finds any new residues, add it
            if new_residues > 0:
                probe_set.add(ligand)
                
                # Update current predictions
                for pname, indices in pred_pockets.items():
                    if pname in current_predictions:
                        current_predictions[pname].update(indices)
                    else:
                        current_predictions[pname] = indices
                
                # Update coverage
                covered_residues += new_residues
                current_coverage = covered_residues / total_residues
                
                print(f"Added ligand {ligand}, found {new_residues} new residues, coverage: {current_coverage:.3f}")
                
                # Check if pocket set is now fully covered or coverage exceeds 80%
                if covered_residues / total_residues >= 0.8:
                    print(f"Pocket set coverage exceeded 80%, stopping this batch")
                    break
        
        coverage_history.append({
            'batch': current_batch + 1,
            'coverage': current_coverage,
            'n_ligands': len(probe_set),
            'total_residues': total_residues,
            'covered_residues': covered_residues,
            'new_proteins': len(batch_proteins),
            'total_proteins': len(current_predictions),
            'skipped': False
        })
        
        current_batch += 1
    
    # Save final probe set to database
    save_probe_set(probe_set)
    
    return probe_set, coverage_history

def plot_coverage_history(coverage_history):
    """Plot coverage history metrics.
    
    Args:
        coverage_history: list of coverage history dictionaries
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Set figure size
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot coverage vs batch number
    batches = [entry['batch'] for entry in coverage_history]
    coverage = [entry['coverage'] for entry in coverage_history]
    plt.plot(batches, coverage, '-o', color='#43a3ef', label='Coverage')
    
    # Customize plot
    plt.xlabel('Batch Number')
    plt.ylabel('Coverage')
    plt.grid(True, linestyle='--', alpha=0.7)
    if plt.gca().get_legend() is not None:
        plt.gca().get_legend().set_frame_on(False)
    
    # Save plot
    print("Saving coverage plot to plots/coverage_history.svg")
    plt.savefig('plots/coverage_history.svg')
    plt.show()
    
    # Plot number of ligands vs batch number
    plt.figure(figsize=(3.5, 2.5))
    n_ligands = [entry['n_ligands'] for entry in coverage_history]
    plt.plot(batches, n_ligands, '-o', color='#ef767b', label='Number of Ligands')
    
    # Customize plot
    plt.xlabel('Batch Number')
    plt.ylabel('Number of Ligands')
    plt.grid(True, linestyle='--', alpha=0.7)
    if plt.gca().get_legend() is not None:
        plt.gca().get_legend().set_frame_on(False)
    
    # Save plot
    print("Saving ligands plot to plots/n_ligands_history.svg")
    plt.savefig('plots/n_ligands_history.svg')
    plt.show()
    
    # Plot residues per ligand
    plt.figure(figsize=(3.5, 2.5))
    residues_per_ligand = [entry['covered_residues']/entry['n_ligands'] for entry in coverage_history]
    plt.plot(batches, residues_per_ligand, '-o', color='#43a3ef', label='Residues per Ligand')
    
    # Customize plot
    plt.xlabel('Batch Number')
    plt.ylabel('Residues per Ligand')
    plt.grid(True, linestyle='--', alpha=0.7)
    if plt.gca().get_legend() is not None:
        plt.gca().get_legend().set_frame_on(False)
    
    # Save plot
    print("Saving residues per ligand plot to plots/residues_per_ligand.svg")
    plt.savefig('plots/residues_per_ligand.svg')
    plt.show()

def visualize_top_probes(n=3):
    """Visualize the first n probes from the probe set table.
    
    Args:
        n: number of probes to visualize
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get first n probes from probe set
        cur.execute("""
            SELECT lname 
            FROM probe_set 
            ORDER BY id 
            LIMIT %s
        """, (n,))
        probe_names = [row[0] for row in cur.fetchall()]
        
        # Get mol data for each probe
        for i, lname in enumerate(probe_names):
            cur.execute("""
                SELECT mol 
                FROM ligands 
                WHERE name = %s
            """, (lname,))
            mol_data = cur.fetchone()[0].tobytes().decode('utf-8')
            
            # Convert mol data to RDKit molecule and remove all hydrogens
            mol = Chem.MolFromMolBlock(mol_data, removeHs=True)
            if mol is None:
                print(f"Failed to parse molecule for {lname}")
                continue
            
            # Generate 2D coordinates first
            AllChem.Compute2DCoords(mol)
            
            # Then remove all hydrogens
            mol = Chem.RemoveAllHs(mol)
                
            # Generate SMILES
            smiles = Chem.MolToSmiles(mol)
            print(f"\nProbe {i+1} ({lname}):")
            print(f"SMILES: {smiles}")
            
            # Draw molecule
            drawer = Draw.rdMolDraw2D.MolDraw2DSVG(350, 250)  # Size matches our 3.5,2.5 ratio
            
            # Customize drawing options
            opts = drawer.drawOptions()
            opts.addStereoAnnotation = False  # Hide R/S stereochemistry annotations
            opts.addAtomIndices = False
            opts.baseFontSize = 1
            opts.bondLineWidth = 3
            opts.noAtomLabels = True  # Hide all atom labels
            
            # Set custom atom colors
            opts.updateAtomPalette({
                7: (0, 0, 1),  # N: blue
                8: (1, 0, 0),  # O: red
                6: (0.4, 0.4, 0.4)  # C: grey
            })
            
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            
            # Save SVG
            svg_path = f'plots/probe_{i+1}_{lname}.svg'
            print(f"Saving structure to {svg_path}")
            with open(svg_path, 'w') as f:
                f.write(svg)
            
            # Display SVG
            display(SVG(svg))

if __name__ == "__main__":
    # Check if results file exists
    if os.path.exists('probe_set_results.pkl'):
        print("Loading existing results...")
        with open('probe_set_results.pkl', 'rb') as f:
            results = pickle.load(f)
        probe_set = results['probe_set']
        coverage_history = results['coverage_history']
        
        # Analyze and plot results
        plot_coverage_history(coverage_history)
        
        # Visualize top probes
        print("\nVisualizing top 3 probes...")
        visualize_top_probes(3)
    else:
        # Set random seed for reproducibility
        random.seed(42)
        
        # Find probe ligands
        probe_set, coverage_history = find_probe_ligands(batch_size=50, max_batches=10)
        
        # Plot results
        plot_coverage_history(coverage_history)
        
        # Save results
        results = {
            'probe_set': probe_set,
            'coverage_history': coverage_history
        }
        with open('probe_set_results.pkl', 'wb') as f:
            pickle.dump(results, f)

# %%


