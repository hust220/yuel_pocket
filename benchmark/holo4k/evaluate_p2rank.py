import os
import csv
import numpy as np
from pathlib import Path
from rdkit import Chem
import matplotlib.pyplot as plt

def calculate_center(atoms):
    """Calculate the geometric center (centroid) of a list of atoms."""
    coords = [atom for atom in atoms]
    if not coords:
        return None
    return np.mean(coords, axis=0)

def calculate_dca(pocket_center, ligand_atom_coords):
    """
    Calculate Distance to Center of Active site (DCA).
    Distance from pocket center to the closest ligand atom.
    """
    if pocket_center is None or len(ligand_atom_coords) == 0:
        return float('inf')
    
    # Vectorized calculation
    ligand_coords = np.array(ligand_atom_coords)
    dists = np.linalg.norm(ligand_coords - pocket_center, axis=1)
    return np.min(dists)

def calculate_dcc(pocket_center, ligand_center):
    """
    Calculate Distance to Center of Center (DCC).
    Distance from pocket center to ligand center.
    """
    if pocket_center is None or ligand_center is None:
        return float('inf')
    return np.linalg.norm(pocket_center - ligand_center)

def load_p2rank_predictions(csv_path):
    """
    Load p2rank predictions from CSV file.
    Returns a list of dictionaries, each containing pocket info.
    """
    predictions = []
    if not os.path.exists(csv_path):
        return predictions

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return predictions
            
        clean_fieldnames = [fn.strip() for fn in fieldnames]
        reader.fieldnames = clean_fieldnames
        
        for row in reader:
            clean_row = {k.strip(): v.strip() for k, v in row.items()}
            
            try:
                cx = float(clean_row['center_x'])
                cy = float(clean_row['center_y'])
                cz = float(clean_row['center_z'])
                score = float(clean_row['score'])
                rank = int(clean_row['rank'])
                
                predictions.append({
                    'rank': rank,
                    'score': score,
                    'center': np.array([cx, cy, cz])
                })
            except KeyError:
                continue

    predictions.sort(key=lambda x: x['rank'])
    return predictions

def get_ligand_coords(lig_file):
    """Load ligand coordinates from SDF or PDB."""
    if str(lig_file).endswith('.sdf'):
        try:
            suppl = Chem.SDMolSupplier(str(lig_file), sanitize=False)
            mol = suppl[0]
            if mol is None: return None
            conf = mol.GetConformer()
            return conf.GetPositions()
        except Exception as e:
            print(f"Error loading SDF {lig_file}: {e}")
            return None
    elif str(lig_file).endswith('.pdb'):
        # Simple PDB parser for coordinates
        coords = []
        try:
            with open(lig_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            pass
            if not coords: return None
            return np.array(coords)
        except Exception as e:
            print(f"Error loading PDB {lig_file}: {e}")
            return None
    return None

import matplotlib.pyplot as plt

def plot_success_rates(cutoffs, top1_rates, top3_rates, title, ylabel, filename):
    plt.figure(figsize=(2.5, 2))
    # Convert rates to percentages
    plt.plot(cutoffs, [r * 100 for r in top1_rates], marker='o', markersize=3, 
             label='Top 1', color='#EF767B', linewidth=1.2)
    plt.plot(cutoffs, [r * 100 for r in top3_rates], marker='o', markersize=3, 
             label='Top 3', color='#43A3EF', linewidth=1.2)
    
    plt.xlabel('Distance Threshold (Å)', fontsize=7)
    plt.ylabel('Success Rate (%)', fontsize=7)
    
    plt.xticks(cutoffs, fontsize=6)
    plt.yticks(range(0, 101, 20), fontsize=6)
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    plt.legend(fontsize=6, loc='lower right', frameon=True)
    plt.tight_layout()
    
    # Save as PNG and SVG
    plt.savefig(filename, dpi=300)
    svg_filename = str(filename).replace('.png', '.svg')
    plt.savefig(svg_filename, dpi=300)
    print(f"Plot saved to {filename} and {svg_filename}")
    plt.close()

def evaluate_predictions(dataset_folder="test50"):
    base_dir = Path(__file__).parent
    p2rank_dir = base_dir / f"{dataset_folder}_p2rank_predictions"
    dataset_dir = base_dir / dataset_folder
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    # Support both .sdf and .pdb
    ligand_files = list(dataset_dir.glob('*_ligand.sdf')) + list(dataset_dir.glob('*_ligand.pdb'))
    if not ligand_files:
        print("No ligand files found.")
        return

    cutoffs = [4, 5, 6, 7, 8, 9, 10]
    all_dca_top1 = []
    all_dca_top3 = []
    all_dcc_top1 = []
    all_dcc_top3 = []
    
    print(f"{'SystemID':<30} {'Top1_DCA':<10} {'Top1_DCC':<10}")
    print("-" * 55)

    total_cases = 0

    for lig_file in ligand_files:
        filename = lig_file.name
        
        # Holo4k logic: 1a0j_BEN_ligand.pdb -> PDBID 1a0j
        # But prediction is 1a0j_protein.pdb_predictions.csv
        
        # Try to guess format
        if len(filename.split('_')) >= 2 and len(filename.split('_')[0]) == 4:
             parts = filename.split('_')
             candidate_pdb_id = parts[0]
             
             # Check if protein file {candidate_pdb_id}_protein.pdb exists
             if (dataset_dir / f"{candidate_pdb_id}_protein.pdb").exists():
                 system_id = candidate_pdb_id
             else:
                  # Fallback
                  if "_ligand" in filename:
                        system_id = filename.split('_ligand')[0]
                  else:
                        system_id = lig_file.stem.replace('.sdf', '').replace('.pdb', '')
        else:
             if "_ligand" in filename:
                system_id = filename.split('_ligand')[0]
             else:
                system_id = lig_file.stem.replace('.sdf', '').replace('.pdb', '')

        # P2Rank output naming convention in this dataset?
        # Usually {system_id}_protein.pdb_predictions.csv
        
        prot_filename = f"{system_id}_protein.pdb"
        pred_csv_name = f"{prot_filename}_predictions.csv"
        pred_csv_path = p2rank_dir / pred_csv_name
        
        if not pred_csv_path.exists():
            continue

        ligand_coords = get_ligand_coords(lig_file)
        if ligand_coords is None:
            continue
            
        ligand_center = np.mean(ligand_coords, axis=0)
        
        preds = load_p2rank_predictions(pred_csv_path)
        if not preds:
            continue
            
        # Top 1 distances
        top1_pocket = preds[0]
        top1_dca = calculate_dca(top1_pocket['center'], ligand_coords)
        top1_dcc = calculate_dcc(top1_pocket['center'], ligand_center)
        
        all_dca_top1.append(top1_dca)
        all_dcc_top1.append(top1_dcc)
        
        # Top 3 distances (minimum distance among top 3)
        top3_preds = preds[:3]
        top3_dcas = [calculate_dca(p['center'], ligand_coords) for p in top3_preds]
        top3_dccs = [calculate_dcc(p['center'], ligand_center) for p in top3_preds]
        
        all_dca_top3.append(min(top3_dcas))
        all_dcc_top3.append(min(top3_dccs))
        
        print(f"{system_id:<30} {top1_dca:<10.2f} {top1_dcc:<10.2f}")
        total_cases += 1
            
    if total_cases > 0:
        print("-" * 55)
        print(f"Total Cases: {total_cases}")
        
        # Calculate success rates for each cutoff
        dca_top1_rates = []
        dca_top3_rates = []
        dcc_top1_rates = []
        dcc_top3_rates = []
        
        print(f"\n{'Cutoff':<10} {'DCA Top1':<10} {'DCA Top3':<10} {'DCC Top1':<10} {'DCC Top3':<10}")
        print("-" * 55)
        
        for c in cutoffs:
            r1_dca = sum(1 for d in all_dca_top1 if d < c) / total_cases
            r3_dca = sum(1 for d in all_dca_top3 if d < c) / total_cases
            r1_dcc = sum(1 for d in all_dcc_top1 if d < c) / total_cases
            r3_dcc = sum(1 for d in all_dcc_top3 if d < c) / total_cases
            
            dca_top1_rates.append(r1_dca)
            dca_top3_rates.append(r3_dca)
            dcc_top1_rates.append(r1_dcc)
            dcc_top3_rates.append(r3_dcc)
            
            print(f"{c:<10} {r1_dca:<10.2%} {r3_dca:<10.2%} {r1_dcc:<10.2%} {r3_dcc:<10.2%}")
        
        # Plotting
        plot_success_rates(cutoffs, dca_top1_rates, dca_top3_rates, 
                          'P2Rank Success Rate (DCA)', 'Success Rate (DCA < Cutoff)', 
                          base_dir / f'p2rank_{dataset_folder}_dca_success.png')
        
        plot_success_rates(cutoffs, dcc_top1_rates, dcc_top3_rates, 
                          'P2Rank Success Rate (DCC)', 'Success Rate (DCC < Cutoff)', 
                          base_dir / f'p2rank_{dataset_folder}_dcc_success.png')
        
        # Save results to CSV
        csv_filename = base_dir / f'p2rank_{dataset_folder}_results.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Cutoff', 'DCA_Top1', 'DCA_Top3', 'DCC_Top1', 'DCC_Top3'])
            for i, c in enumerate(cutoffs):
                writer.writerow([
                    c, 
                    f"{dca_top1_rates[i]:.4f}", 
                    f"{dca_top3_rates[i]:.4f}", 
                    f"{dcc_top1_rates[i]:.4f}", 
                    f"{dcc_top3_rates[i]:.4f}"
                ])
        print(f"Success rates saved to {csv_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate p2rank predictions")
    parser.add_argument("folder_name", nargs="?", default="test50", help="Dataset folder name (default: test50)")
    args = parser.parse_args()
    
    evaluate_predictions(args.folder_name)
