# %%

import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.join(__file__, '../../..'))
from src.db_utils import db_connection
from src.clustering import cluster_pocket_predictions
from src.pdb_utils import Structure
from io import StringIO

def add_allosteric_clusters_column():
    """Add allosteric_clusters column to allobench table."""
    from src.db_utils import add_column
    add_column('allobench', 'allosteric_clusters', 'INTEGER[]')

def calculate_allosteric_clusters():
    """Calculate clusters for allosteric sites and store them in allobench table."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get proteins with allosteric sites
        cur.execute("""
            SELECT id, pdb_id, pdb_content, in_allosteric_site
            FROM allobench
            WHERE in_allosteric_site IS NOT NULL 
            AND modulator_class = 'Lig'
            AND array_length(in_allosteric_site, 1) < 2000
        """)
        
        for row in tqdm(cur.fetchall(), desc="Processing proteins"):
            allobench_id, pdb_id, pdb_content, in_allosteric = row
            
            try:
                # Convert site indicators to numpy array
                in_allosteric = np.array(in_allosteric)
                
                # Create Structure object
                structure = Structure()
                structure.read(StringIO(pdb_content))
                
                # Perform clustering
                clusters = cluster_pocket_predictions(in_allosteric, structure)
                
                # Convert numpy array to Python list for database storage
                clusters_list = clusters.tolist()
                
                # Update allobench table with cluster assignments
                cur.execute("""
                    UPDATE allobench 
                    SET allosteric_clusters = %s
                    WHERE id = %s
                """, (clusters_list, allobench_id))
                conn.commit()
                    
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                continue
    
    print("Finished calculating and storing allosteric clusters")

def plot_allosteric_clusters():
    """Plot the distribution of allosteric site cluster sizes."""
    # Get cluster assignments from database
    cluster_sizes = []
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT allosteric_clusters 
            FROM allobench 
            WHERE allosteric_clusters IS NOT NULL
        """)
        
        for row in cur.fetchall():
            clusters = np.array(row[0])
            # Calculate sizes of each cluster (excluding noise points labeled as -1)
            unique_clusters = np.unique(clusters)
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # Exclude noise points
                    cluster_size = np.sum(clusters == cluster_id)
                    cluster_sizes.append(int(cluster_size))
    
    if not cluster_sizes:
        print("No valid clusters found in database!")
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot distribution
    plt.figure(figsize=(3.5, 2.5))
    plt.hist(cluster_sizes, bins='auto', color='#43a3ef', alpha=0.7,
             edgecolor='black', linewidth=1)
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    
    # Print filename before saving
    filename = 'plots/allosteric_clusters.svg'
    print(f"\nSaving plot to {filename}")
    
    # Save and show plot
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    # Print statistics
    print("\nCluster Statistics:")
    print(f"Total number of clusters: {len(cluster_sizes)}")
    print(f"Mean cluster size: {np.mean(cluster_sizes):.1f} ± {np.std(cluster_sizes):.1f}")
    print(f"Median cluster size: {np.median(cluster_sizes):.1f}")
    print(f"Min cluster size: {np.min(cluster_sizes)}")
    print(f"Max cluster size: {np.max(cluster_sizes)}")

if __name__ == "__main__":

    add_allosteric_clusters_column()

    # Calculate and store allosteric clusters
    calculate_allosteric_clusters()
    
    # Plot allosteric clusters
    plot_allosteric_clusters() 

# %%
