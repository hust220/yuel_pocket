import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_plinder_test_set():
    # Detect project root (assume this script is in project_root/benchmark/plinder/)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    
    # 1. Load the data
    parquet_path = project_root / 'data/plinder/data/2024-06/v2/splits/split.parquet'
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        # Fallback to relative path if run from root
        parquet_path = Path('./data/plinder/data/2024-06/v2/splits/split.parquet')
        if not parquet_path.exists():
            return
    
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # 2. Filter for test set
    test_df = df[df['split'] == 'test'].copy()
    print(f"Total systems in test set: {len(test_df)}")
    
    # 3. Extract PDB IDs
    # system_id format: PDBID__Model__ChainInfo
    test_df['pdb_id'] = test_df['system_id'].apply(lambda x: x.split('__')[0])
    
    # 4. Count systems per PDB ID
    counts = test_df.groupby('pdb_id').size().reset_index(name='num_ligands')
    
    # 5. Summary Statistics
    num_unique_pdbs = len(counts)
    print(f"Number of unique PDB IDs in test set: {num_unique_pdbs}")
    
    distribution = counts['num_ligands'].value_counts().sort_index()
    print("\nDistribution of ligands per protein:")
    print(distribution)
    
    pct_multiple = (counts['num_ligands'] > 1).sum() / num_unique_pdbs * 100
    print(f"\nPercentage of PDBs with multiple ligands: {pct_multiple:.2f}%")
    
    # 6. Plotting
    sns.set_theme(style="whitegrid")
    # Using the premium style from previous plots
    plt.figure(figsize=(3.5, 2.5))
    
    plot_data = distribution.reset_index()
    plot_data.columns = ['Ligands per PDB', 'Count']
    
    # Group long tail
    max_to_show = 6
    if plot_data['Ligands per PDB'].max() > max_to_show:
        tail_sum = plot_data[plot_data['Ligands per PDB'] >= max_to_show]['Count'].sum()
        plot_data = plot_data[plot_data['Ligands per PDB'] < max_to_show].copy()
        plot_data = pd.concat([plot_data, pd.DataFrame({'Ligands per PDB': [f'{max_to_show}+'], 'Count': [tail_sum]})])
    
    # Colors matching the theme: #EF767B, #43A3EF, and variations
    # We'll use a gradient or the specific colors
    custom_palette = sns.color_palette("blend:#EF767B,#43A3EF", len(plot_data))
    
    ax = sns.barplot(x='Ligands per PDB', y='Count', data=plot_data, palette=custom_palette)
    
    # Add labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 5), 
                        textcoords = 'offset points',
                        fontsize=6, fontweight='bold')

    plt.xlabel('Ligands per Protein', fontsize=7)
    plt.ylabel('Number of Proteins', fontsize=7)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    
    plt.tight_layout()
    output_path = script_path.parent / 'plinder_test_ligand_distribution.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    
    # Also save a small CSV for reference
    csv_path = script_path.parent / 'plinder_test_pdb_counts.csv'
    counts.to_csv(csv_path, index=False)
    print(f"Detailed counts saved to {csv_path}")

if __name__ == "__main__":
    analyze_plinder_test_set()
