
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_rmsd_distribution(csv_path, output_path):
    # Load data
    df = pd.read_csv(csv_path)
    rmsd_values = df['RMSD'].values
    
    # Filter out very large values for better visualization if needed, 
    # but let's show up to a reasonable limit or log scale?
    # Given the stats (avg 1.5, max 42), most are small. 
    # A histogram with a cut-off or log scale might be good, 
    # but I'll stick to a standard histogram first, maybe zoomed in.
    
    # Let's clip the display to max 10A or 20A to see the distribution clearly, 
    # or just plot all. Let's filter for the plot range but report stats on all.
    # Actually, let's keep it simple and just plot.
    
    # Plotting style from evaluate_yuelpocket2.py
    plt.figure(figsize=(2.5, 2))
    
    # Histogram
    # Using the blue color from the reference script
    color = '#43A3EF' 
    
    # Create bins. 
    # Most data is probably < 5A. 
    # Let's defined bins up to 10A, and maybe a catch-all bin? 
    # Or just standard auto bins.
    # Let's use a range that covers most data, e.g., 0 to 10.
    n, bins, patches = plt.hist(rmsd_values, bins=50, range=(0, 10), 
                                color=color, edgecolor='none', alpha=0.8)
    
    plt.xlabel('RMSD ($\AA$)', fontsize=7)
    plt.ylabel('Count', fontsize=7)
    
    # Ticks formatting
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    
    # Grid
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    
    # Layout
    plt.tight_layout()
    
    # Save
    plt.savefig(str(output_path).replace('.png', '.svg'), dpi=300)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_path} and SVG version.")
    
    # Print some stats for the user
    print(f"RMSD Statistics:")
    print(f"Total systems: {len(rmsd_values)}")
    print(f"RMSD < 1A: {np.sum(rmsd_values < 1.0)} ({np.sum(rmsd_values < 1.0)/len(rmsd_values):.1%})")
    print(f"RMSD < 2A: {np.sum(rmsd_values < 2.0)} ({np.sum(rmsd_values < 2.0)/len(rmsd_values):.1%})")
    print(f"RMSD < 5A: {np.sum(rmsd_values < 5.0)} ({np.sum(rmsd_values < 5.0)/len(rmsd_values):.1%})")

if __name__ == "__main__":
    csv_file = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/rmsd_test1036_af.csv"
    output_file = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/rmsd_distribution_test1036.png"
    
    plot_rmsd_distribution(csv_file, output_file)
