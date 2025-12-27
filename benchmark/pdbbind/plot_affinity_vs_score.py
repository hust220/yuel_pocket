import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import re
import math
import numpy as np
from pathlib import Path

# Try importing scipy
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Spearman correlation will be skipped.")

def parse_p_val(s):
    # s format: "Ki=0.43uM", "Kd~10nM", "IC50=0.5mM"
    # Returns -log10(Molar)
    
    # Clean string
    s = s.replace('=', '').replace('~', '').replace('<', '').replace('>', '').strip()
    
    # Regex for number and unit
    # Matches: 1.23, 10, 0.5
    # Units: mM, uM, nM, pM, fM
    match = re.search(r'([\d\.]+)\s*(mM|uM|nM|pM|fM)', s)
    if not match:
        return None
    
    val_str = match.group(1)
    unit_str = match.group(2)
    
    try:
        val = float(val_str)
    except ValueError:
        return None
        
    if val <= 0: return None
    
    multiplier = 1.0
    if unit_str == 'mM': multiplier = 1e-3
    elif unit_str == 'uM': multiplier = 1e-6
    elif unit_str == 'nM': multiplier = 1e-9
    elif unit_str == 'pM': multiplier = 1e-12
    elif unit_str == 'fM': multiplier = 1e-15
    
    molar = val * multiplier
    return -math.log10(molar)

def main():
    root = Path('/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/pdbbind')
    pred_dir = root / 'test500_yuelpocket_residues2_predictions'
    index_file = root / 'INDEX_general_PL.2020R1.lst'
    
    if not pred_dir.exists():
        print(f"Directory not found: {pred_dir}")
        return
        
    # 1. Load Experimental Data
    print(f"Reading {index_file}...")
    pdb_pk = {}
    with open(index_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 4:
                pdb_id = parts[0].lower()
                aff = parts[3]
                pk = parse_p_val(aff)
                if pk is not None:
                    pdb_pk[pdb_id] = pk
                    
    print(f"Loaded valid affinity for {len(pdb_pk)} PDBs.")
    
    # 2. Load Predictions
    print(f"Reading predictions from {pred_dir}...")
    files = list(pred_dir.glob('*_predictions.txt'))
    print(f"Found {len(files)} prediction files.")
    
    x = [] # Exp pKd
    y = [] # Pred Score
    
    for p_file in files:
        pdb = p_file.name.replace('_predictions.txt', '').lower()
        if pdb not in pdb_pk:
            continue
            
        try:
            with open(p_file, 'r') as f:
                line = f.readline()
                # Format: # Overall Binding Probability: -3.844080
                if "Overall Binding Probability" in line:
                    val_str = line.split(':')[-1].strip()
                    score = float(val_str)
                    
                    x.append(pdb_pk[pdb])
                    y.append(score)
        except Exception as e:
            print(f"Error reading {p_file.name}: {e}")
            
    n = len(x)
    print(f"Matched {n} systems.")
    
    if n < 2:
        print("Not enough points for correlation/plot.")
        return
        
    # 3. Plot
    title_str = f"Affinity Correlation (N={n})"
    
    if HAS_SCIPY:
        rp, pp = pearsonr(x, y)
        rs, ps = spearmanr(x, y)
        print(f"Pearson r: {rp:.4f}")
        print(f"Spearman r: {rs:.4f}")
        title_str += f"\nPearson r={rp:.3f}, Spearman r={rs:.3f}"
    else:
        # Fallback Pearson using numpy
        rp = np.corrcoef(x, y)[0, 1]
        print(f"Pearson r: {rp:.4f}")
        title_str += f"\nPearson r={rp:.3f}"
        
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.6, s=25, c='#43A3EF', edgecolors='none')
    
    # Regression line
    m, b = np.polyfit(x, y, 1)
    range_x = np.array([min(x), max(x)])
    plt.plot(range_x, m*range_x + b, 'r--', alpha=0.8, label='Fit')
    
    plt.xlabel('Experimental pKd/pKi/pIC50')
    plt.ylabel('Predicted Score (Overall Binding Probability)')
    plt.title(title_str)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = root / 'test500_affinity_scatter.png'
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    main()
