
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from scipy.spatial import KDTree

# Import utils from project
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from src import pdb_utils

def load_ligand_center(lig_file):
    if str(lig_file).endswith('.sdf'):
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(lig_file), sanitize=False)
        mol = suppl[0]
        if mol is None: return None
        coords = mol.GetConformer().GetPositions()
        return np.mean(coords, axis=0)
    elif str(lig_file).endswith('.pdb'):
        structure = pdb_utils.Structure(str(lig_file))
        atoms = structure.get_atoms()
        coords = np.array([atom.get_coord() for atom in atoms])
        if len(coords) == 0: return None
        return np.mean(coords, axis=0)
    return None

def calculate_dca(pocket_center, ligand_coords):
    # This implies we need all ligand coords, not just center, 
    # but the prompt asks for "Distance Threshold 4". 
    # Usually "DCA" (Distance to Closest Atom) is used.
    # evaluate_yuelpocket2.py uses DCA: min(dist(pocket_center, lig_atoms))
    pass

def load_ligand_coords(lig_file):
    if str(lig_file).endswith('.sdf'):
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(lig_file), sanitize=False)
        mol = suppl[0]
        if mol is None: return None
        coords = mol.GetConformer().GetPositions()
        return coords
    elif str(lig_file).endswith('.pdb'):
        structure = pdb_utils.Structure(str(lig_file))
        atoms = structure.get_atoms()
        coords = np.array([atom.get_coord() for atom in atoms])
        if len(coords) == 0: return None
        return coords
    return None

def load_residue_predictions(pred_txt):
    if not pred_txt.exists(): return {}
    preds = {}
    try:
        df = pd.read_csv(pred_txt) # Assuming standard csv format from previous tools
        # The file format in evaluate_yuelpocket2.py used pd.read_csv with comment='#'
        # And keys were (Chain, ResID)
        df = pd.read_csv(pred_txt, comment='#')
        for _, row in df.iterrows():
            chain = row['Chain'] if not pd.isna(row['Chain']) else ' '
            resid = int(row['ResID'])
            score = row['PocketProbability']
            preds[(chain, resid)] = score
    except Exception as e:
        print(f"Error loading {pred_txt}: {e}")
    return preds

ALLOWED_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'SEC', 'PYL', 'ASX', 'GLX', 'XLE'
}

def get_residue_coords(prot_file):
    coords = {}
    structure = pdb_utils.Structure(str(prot_file), skip_hetatm=True)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.res_name in ALLOWED_AMINO_ACIDS: # Filter standard AA
                    res_coords = residue.get_coords()
                    if len(res_coords) > 0:
                        coords[(chain.chain_id, residue.res_id)] = res_coords
        break
    return coords

def analyze_and_plot(rmsd_csv, predictions_dir, dataset_dir, output_file):
    print(f"Loading RMSD from {rmsd_csv}...")
    rmsd_df = pd.read_csv(rmsd_csv)
    
    print(f"Processing predictions from {predictions_dir}...")
    pred_path = Path(predictions_dir)
    data_path = Path(dataset_dir)
    
    results = []
    
    for _, row in rmsd_df.iterrows():
        system_id = row['SystemID']
        rmsd = row['RMSD']
        
        # Files
        pred_txt = pred_path / f"{system_id}_predictions.txt"
        
        ligand_sdf = data_path / f"{system_id}_ligand.sdf"
        ligand_pdb = data_path / f"{system_id}_ligand.pdb"
        ligand_file = ligand_sdf if ligand_sdf.exists() else (ligand_pdb if ligand_pdb.exists() else None)
        
        prot_file = data_path / f"{system_id}_protein.pdb"
        
        if not pred_txt.exists() or not ligand_file or not prot_file.exists():
            continue
            
        # Load Data
        pred_scores = load_residue_predictions(pred_txt)
        res_coords_dict = get_residue_coords(prot_file)
        lig_coords = load_ligand_coords(ligand_file)
        
        if not pred_scores or not res_coords_dict or lig_coords is None:
            continue
            
        # Calculate Distances and Identify Pocket Residues
        # We need to sort all residues by score and check if any of top K is a pocket residue
        # Pocket residue definition: min dist to ligand < threshold (e.g. 4.0A as requested)
        
        system_residues = []
        lig_tree = KDTree(lig_coords)
        
        threshold = 4.0 # Distance Threshold
        rank_cutoff = 3 # Rank Threshold
        
        for key, coords in res_coords_dict.items():
            # Get max score for this residue (if multiple entries, usually one)
            score = pred_scores.get(key, -1.0)
            if score == -1.0:
                # Prediction might use ' ' for chain but PDB has 'A', or vice versa
                # Try simple variations if needed, but let's stick to strict first
                # Actually evaluate_yuelpocket2 handles this carefully.
                # Let's try to match logic. 
                # If key not found, skip or treat as 0? Usually we only care about predicted ones.
                continue
            
            dists, _ = lig_tree.query(coords)
            min_dist = np.min(dists)
            system_residues.append((score, min_dist))
            
        # Sort by score descending
        system_residues.sort(key=lambda x: x[0], reverse=True)
        
        # Check Success
        # Success if ANY residue within top K (Rank <= K) has dist < threshold
        # Actually evaluate_yuelpocket2 logic:
        # "for rank, (is_pocket, _) in enumerate(residue_data, start=1): if is_pocket: best_rank = rank; break"
        # So we look for the Rank of the FIRST true pocket residue.
        # If that Rank <= 3, then it's a success for "Rank <= 3".
        
        # Note: evaluate_yuelpocket2 logic for cutoff curve uses:
        # "Is the rank of the first residue with dist < T less than or equal to K?"
        
        is_success = 0
        current_rank = 0
        found = False
        
        for i, (score, dist) in enumerate(system_residues):
            rank = i + 1
            if dist < threshold:
                if rank <= rank_cutoff:
                    is_success = 1
                found = True
                break # Found the highest scoring pocket residue
        
        # If no pocket residue found at all (very unlikely), success is 0.
        
        results.append({
            'SystemID': system_id,
            'RMSD': rmsd,
            'Success': is_success
        })
        
    if not results:
        print("No matching data found.")
        return

    df = pd.DataFrame(results)
    
    # Save extended CSV
    output_csv = dataset_dir.parent / "rmsd_success_test1036_af.csv"
    df.to_csv(output_csv, index=False)
    print(f"Extended results saved to {output_csv}")
    
    # Print high RMSD systems
    high_rmsd = df[df['RMSD'] > 10]
    if not high_rmsd.empty:
        print("\nSystems with RMSD > 10:")
        print(high_rmsd[['SystemID', 'RMSD']].to_string(index=False))
    
    # Create Bins for Scatter Plot
    # We want "Success Rate" vs "RMSD".
    # Adaptive binning or fixed?
    # Let's try flexible bins based on data quantiles or fixed width?
    # Fixed width is easier to interpret: e.g. 0.5A bins
    
    # Determine bins
    min_rmsd = df['RMSD'].min()
    max_rmsd = df['RMSD'].max()
    
    # Use bins of width 0.5A, or if range is small, 0.2A
    # For test50, ranges were ~0.2 to 4.0.
    # Let's use 5-10 bins total.
    
    # Let's define bins manually for clarity or use qcut
    # Using fixed bins [0, 0.5, 1.0, 1.5, 2.0, 3.0, >3.0] might be good.
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 100.0]
    labels = [0.25, 0.75, 1.25, 1.75, 2.5, 4.0, 6.0] # Approximate centers for plotting
    
    # Actually, calculating mean RMSD of the bin is better for X-axis
    
    df['Bin'] = pd.cut(df['RMSD'], bins=bins)
    
    summary = df.groupby('Bin').agg(
        SuccessRate=('Success', 'mean'),
        RMSDMean=('RMSD', 'mean'),
        Count=('Success', 'count')
    ).reset_index()
    
    # Filter empty bins
    summary = summary[summary['Count'] > 0]
    
    print("Summary:")
    print(summary)
    
    # Plot
    plt.figure(figsize=(2.5, 2))
    
    # Plot binned averages (Success Rate vs RMSD Bin Center/Mean)
    # Using marker='o' to show the 6 points representing the bins
    # Blue color: #43A3EF
    plt.plot(summary['RMSDMean'], summary['SuccessRate']*100, 
             color='#43A3EF', marker='o', markersize=4, 
             linestyle='-', linewidth=1.2, label='Success Rate')
    
    plt.xlabel('RMSD ($\AA$)', fontsize=7)
    plt.ylabel('Success Rate (%)', fontsize=7)
    
    # Ticks
    plt.xticks(fontsize=6)
    plt.yticks(range(0, 101, 20), fontsize=6)
    plt.ylim(-5, 105)
    
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    plt.tight_layout()
    
    output_str = str(output_file)
    ext = output_str.split('.')[-1]
    base = output_str.replace(f'.{ext}', '')
    
    plt.savefig(f"{base}.png", dpi=300)
    plt.savefig(f"{base}.svg", dpi=300)
    plt.close()
    print(f"Plots saved to {base}.png and {base}.svg")

if __name__ == "__main__":
    # Defaults
    base_dir = Path("/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold")
    
    # Updated to test1036 dataset
    rmsd_csv = base_dir / "rmsd_test1036_af.csv"
    pred_dir = base_dir / "test1036_af_yuelpocket_residues2_predictions"
    data_dir = base_dir / "test1036_af"
    output = base_dir / "success_rate_vs_rmsd_test1036.png"
    
    analyze_and_plot(rmsd_csv, pred_dir, data_dir, output)
