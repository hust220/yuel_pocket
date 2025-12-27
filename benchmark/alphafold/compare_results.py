import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_comparison(csv_files, labels, output_dir, prefix="comparison"):
    """
    Compare multiple evaluation CSV files on the same plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataframes = []
    for f in csv_files:
        df = pd.read_csv(f)
        dataframes.append(df)
    
    # Custom color palette matching the project theme
    # Red, Blue, Orange, Green, Purple, etc.
    custom_colors = ['#EF767B', '#43A3EF', '#F5A623', '#7ED321', '#9013FE', '#4A90E2']
    
    # Combined Plots (DCA & DCC)
    # DCA Top 1 & 3 for all models
    for base_metric in ['DCA', 'DCC']:
        plt.figure(figsize=(2.5, 2))
        
        for i, (df, label) in enumerate(zip(dataframes, labels)):
            # Use cyclic colors if more models than colors
            color = custom_colors[i % len(custom_colors)]
            
            # Top 1 - solid line
            # Assuming data in CSV is 0.0-1.0, convert to percentage
            # Check first value to be sure, usually these CSVs have 0-1
            
            # Top 1
            plt.plot(df['Cutoff'], df[f'{base_metric}_Top1'] * 100, 
                    marker='o', markersize=3, label=f'{label} (Top 1)', 
                    color=color, linewidth=1.2)
            # Top 3 - dashed line
            plt.plot(df['Cutoff'], df[f'{base_metric}_Top3'] * 100, 
                    marker='o', markersize=3, label=f'{label} (Top 3)', 
                    color=color, linestyle='--', alpha=0.7, linewidth=1.2)
        
        # plt.title(f'{base_metric} Success Rates', fontsize=7) # Title often omitted in this style
        plt.xlabel('Distance Cutoff (Å)', fontsize=7)
        plt.ylabel('Success Rate (%)', fontsize=7)
        
        if not dataframes:
            continue
            
        plt.xticks(dataframes[0]['Cutoff'], fontsize=6)
        plt.yticks(range(0, 101, 20), fontsize=6)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
        plt.legend(fontsize=5, loc='lower right', frameon=True) # Smaller font for legend as it can get crowded
        plt.tight_layout()
        
        out_path = output_dir / f"{prefix}_{base_metric.lower()}_combined_comparison.png"
        out_svg_path = output_dir / f"{prefix}_{base_metric.lower()}_combined_comparison.svg"
        
        plt.savefig(out_path, dpi=300)
        plt.savefig(out_svg_path, dpi=300)
        print(f"Saved combined plot to {out_path} and {out_svg_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare pocket prediction results from CSV files")
    parser.add_argument("--csvs", nargs="+", required=True, help="List of CSV files to compare")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for each CSV file in the plot")
    parser.add_argument("--output_dir", default=".", help="Directory to save comparison plots")
    parser.add_argument("--prefix", default="pocket_comp", help="Prefix for output filenames")
    
    args = parser.parse_args()
    
    if len(args.csvs) != len(args.labels):
        print("Error: Number of CSV files must match number of labels.")
        exit(1)
        
    plot_comparison(args.csvs, args.labels, args.output_dir, args.prefix)
