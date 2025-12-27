import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_scores(txt_path):
    scores = []
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            # Skip header if present
            start_idx = 0
            if lines and ("Score" in lines[0] or "X" in lines[0]):
                start_idx = 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        scores.append(float(parts[3]))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
    return np.array(scores)

def plot_single_distribution(scores, filename, output_path):
    plt.figure(figsize=(6, 6))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({'Score': scores})
    
    # Box plot
    sns.boxplot(y='Score', data=df, color='lightblue', showfliers=False)
    
    # Swarm plot (or Strip plot if too many points)
    if len(scores) < 1000:
        sns.swarmplot(y='Score', data=df, color='black', alpha=0.6, size=3)
    else:
        # Fallback to strip plot for performance
        sns.stripplot(y='Score', data=df, color='black', alpha=0.4, size=2, jitter=True)
        plt.title(f"{filename}\n(Strip plot used due to >1000 points)")

    plt.title(f"Score Distribution: {filename}")
    plt.ylabel("Score")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def plot_combined_histogram(all_scores, output_path):
    plt.figure(figsize=(10, 6))
    
    sns.histplot(all_scores, bins=50, kde=True, color='skyblue', edgecolor='black')
    
    plt.title("Combined Score Histogram (All SAS Points)")
    plt.xlabel("Score")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    txt_files = list(folder.glob("*_predictions.txt"))
    if not txt_files:
        print("No prediction txt files found.")
        return

    print(f"Found {len(txt_files)} prediction files in {folder}")
    
    all_scores = []
    
    for txt_file in txt_files:
        scores = load_scores(txt_file)
        if len(scores) == 0:
            continue
            
        all_scores.extend(scores)
        
        # Plot individual Box+Swarm
        # Extract protein name for title: 1a4q_protein.pdb_predictions.txt -> 1a4q
        # Or just use full filename
        simple_name = txt_file.name.split('_')[0] 
        output_png = txt_file.with_suffix('.png') # .txt -> .png
        if output_png.exists():
             # Optional: skip if exists, but user might want to overwrite
             pass
        
        print(f"Plotting {txt_file.name} ({len(scores)} points)...")
        plot_single_distribution(scores, simple_name, output_png)
        
    # Plot Combined Histogram
    if all_scores:
        hist_path = folder / "all_scores_histogram.png"
        print(f"Plotting combined histogram to {hist_path}...")
        plot_combined_histogram(np.array(all_scores), hist_path)
    else:
        print("No scores found across all files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot YuelPocket score distributions")
    parser.add_argument("folder", help="Folder containing _predictions.txt files")
    args = parser.parse_args()
    
    main(args.folder)
