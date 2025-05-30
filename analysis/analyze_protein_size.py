#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from src.db_utils import db_connection
import numpy as np

def get_protein_sizes():
    """Get size of all proteins from the database"""
    with db_connection('moad') as conn:
        query = """
        SELECT name, size 
        FROM proteins 
        WHERE size IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
    return df

def plot_size_distribution(df, save_path=None):
    """Plot the distribution of protein sizes
    
    Args:
        df: DataFrame containing protein sizes
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 5))
    
    # Create main histogram with KDE
    sns.histplot(data=df, x='size', bins=50, kde=True)
    
    # Add mean and median lines
    mean_size = df['size'].mean()
    median_size = df['size'].median()
    
    plt.axvline(mean_size, color='red', linestyle='--', label=f'Mean: {mean_size:.1f}')
    plt.axvline(median_size, color='green', linestyle='--', label=f'Median: {median_size:.1f}')
    
    # Add labels and title
    plt.xlabel('Number of Residues')
    plt.ylabel('Count')
    plt.title('Distribution of Protein Sizes')
    plt.legend()
    
    # Add summary statistics as text
    stats_text = f"""
    Total proteins: {len(df):,}
    Mean size: {mean_size:.1f}
    Median size: {median_size:.1f}
    Min size: {df['size'].min():.1f}
    Max size: {df['size'].max():.1f}
    Std dev: {df['size'].std():.1f}
    """
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_cumulative_distribution(df, save_path=None):
    """Plot the cumulative distribution of protein sizes
    
    Args:
        df: DataFrame containing protein sizes
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 5))
    
    # Sort the data
    sorted_sizes = np.sort(df['size'])
    
    # Calculate cumulative probabilities
    y = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    
    # Plot the cumulative distribution
    plt.plot(sorted_sizes, y, linewidth=2)
    
    # Add mean and median lines
    mean_size = df['size'].mean()
    median_size = df['size'].median()
    
    plt.axvline(mean_size, color='red', linestyle='--', label=f'Mean: {mean_size:.1f}')
    plt.axvline(median_size, color='green', linestyle='--', label=f'Median: {median_size:.1f}')
    
    # Add labels and title
    plt.xlabel('Number of Residues')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Protein Sizes')
    plt.legend()
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

#%%

# Get protein sizes
df = get_protein_sizes()

# Print basic statistics
print("\nProtein Size Statistics:")
print(f"Total proteins: {len(df):,}")
print(f"Mean size: {df['size'].mean():.1f} residues")
print(f"Median size: {df['size'].median():.1f} residues")
print(f"Min size: {df['size'].min():.1f} residues")
print(f"Max size: {df['size'].max():.1f} residues")
print(f"Standard deviation: {df['size'].std():.1f} residues")

# Plot distributions
# plot_size_distribution(df, save_path='protein_size_distribution.png')
plot_cumulative_distribution(df, save_path='protein_size_cumulative.png')

# %%
