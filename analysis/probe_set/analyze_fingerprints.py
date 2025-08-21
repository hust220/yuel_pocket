#%%

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
sys.path.append('../../')
from src.db_utils import db_connection
import pickle
import hdbscan
from sklearn.preprocessing import StandardScaler

# Create figures directory if it doesn't exist
FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

def load_fingerprint_data():
    """Load fingerprint data from database into a pandas DataFrame."""
    with db_connection() as conn:
        # Get probe columns
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'probe_fingerprints' 
            AND column_name LIKE 'probe_%'
            ORDER BY column_name
        """)
        probe_columns = [row[0] for row in cur.fetchall()]
        
        # Load data
        query = f"""
            SELECT pname, fingerprint, {', '.join(probe_columns)}
            FROM probe_fingerprints
        """
        df = pd.read_sql_query(query, conn)
        
        # Convert fingerprint binary to numpy arrays
        df['fingerprint_array'] = df['fingerprint'].apply(pickle.loads)
        
        return df, probe_columns

def analyze_probe_importance(df, probe_columns):
    """Analyze which probes are most distinctive."""
    # Calculate variance of each probe
    variances = df[probe_columns].var()
    
    # Calculate correlation between probes
    correlations = df[probe_columns].corr()
    
    # Plot probe variances
    plt.figure(figsize=(12, 6))
    variances.sort_values(ascending=False).plot(kind='bar')
    plt.title('Probe Distinctiveness (Variance)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(FIGURE_DIR, 'probe_variance.png'))
    plt.close()
    
    # Plot probe correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, cmap='coolwarm', center=0)
    plt.title('Probe Correlations')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(FIGURE_DIR, 'probe_correlations.png'))
    plt.close()
    
    return variances, correlations

def cluster_proteins_minibatch(df, probe_columns, n_clusters=50, batch_size=1000):
    """Cluster proteins using MiniBatchKMeans for memory efficiency."""
    print("Performing MiniBatch K-means clustering...")
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[probe_columns].values)
    
    # Perform clustering
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                            batch_size=batch_size,
                            random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Save cluster centers for analysis
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=probe_columns
    )
    centers.to_csv(os.path.join(FIGURE_DIR, 'cluster_centers.csv'))
    
    # Visualize cluster sizes
    plt.figure(figsize=(12, 6))
    pd.Series(clusters).value_counts().plot(kind='bar')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Proteins')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(FIGURE_DIR, 'cluster_sizes.png'))
    plt.close()
    
    return clusters, centers

def cluster_proteins_hdbscan(df, probe_columns, min_cluster_size=50):
    """Cluster proteins using HDBSCAN for automatic density-based clustering."""
    print("Performing HDBSCAN clustering...")
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[probe_columns].values)
    
    # Perform clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=5,
                               metric='euclidean',
                               core_dist_n_jobs=-1)
    clusters = clusterer.fit_predict(X)
    
    # Visualize cluster sizes (excluding noise points which are labeled as -1)
    plt.figure(figsize=(12, 6))
    cluster_sizes = pd.Series(clusters[clusters >= 0]).value_counts()
    cluster_sizes.plot(kind='bar')
    plt.title('Cluster Size Distribution (Excluding Noise)')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Proteins')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(FIGURE_DIR, 'hdbscan_cluster_sizes.png'))
    plt.close()
    
    # Print noise point percentage
    noise_percent = (clusters == -1).sum() / len(clusters) * 100
    print(f"Percentage of proteins classified as noise: {noise_percent:.2f}%")
    
    return clusters

def cluster_proteins_birch(df, probe_columns, n_clusters=50, threshold=0.5):
    """Cluster proteins using BIRCH for memory-efficient hierarchical clustering."""
    print("Performing BIRCH clustering...")
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[probe_columns].values)
    
    # Perform clustering
    brc = Birch(n_clusters=n_clusters, threshold=threshold)
    clusters = brc.fit_predict(X)
    
    # Visualize cluster sizes
    plt.figure(figsize=(12, 6))
    pd.Series(clusters).value_counts().plot(kind='bar')
    plt.title('BIRCH Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Proteins')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(FIGURE_DIR, 'birch_cluster_sizes.png'))
    plt.close()
    
    return clusters

def find_similar_proteins_efficient(df, probe_columns, n_neighbors=5, batch_size=1000):
    """Find similar proteins using batched processing to save memory."""
    similar_pairs = []
    protein_names = df['pname'].values
    X = df[probe_columns].values
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Finding similar proteins"):
        batch_start = i
        batch_end = min(i + batch_size, len(df))
        
        # Calculate distances for this batch
        batch_distances = np.zeros((batch_end - batch_start, len(df)))
        for j in range(batch_start, batch_end):
            batch_distances[j - batch_start] = np.sum((X - X[j])**2, axis=1)
        
        # Find nearest neighbors for each protein in batch
        for j in range(batch_distances.shape[0]):
            protein_idx = batch_start + j
            # Get indices of nearest neighbors (excluding self)
            nearest = np.argpartition(batch_distances[j], n_neighbors+1)[:n_neighbors+1]
            nearest = nearest[nearest != protein_idx][:n_neighbors]
            
            for neighbor_idx in nearest:
                similar_pairs.append({
                    'protein1': protein_names[protein_idx],
                    'protein2': protein_names[neighbor_idx],
                    'distance': batch_distances[j, neighbor_idx]
                })
    
    # Convert to DataFrame and sort by distance
    similar_df = pd.DataFrame(similar_pairs)
    similar_df = similar_df.sort_values('distance')
    similar_df.to_csv(os.path.join(FIGURE_DIR, 'similar_proteins.csv'), index=False)
    return similar_df

def visualize_pca_incremental(df, probe_columns, batch_size=1000):
    """Visualize proteins using Incremental PCA for memory efficiency."""
    # Initialize Incremental PCA
    ipca = IncrementalPCA(n_components=2)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[probe_columns].values)
    
    # Fit IPCA in batches
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        ipca.partial_fit(batch)
    
    # Transform all data
    X_pca = ipca.transform(X)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    plt.title('PCA of Protein Fingerprints (Incremental)')
    plt.xlabel(f'PC1 ({ipca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({ipca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Add some protein labels
    for i in range(0, len(df), len(df)//20):  # Label ~20 points
        plt.annotate(df['pname'].iloc[i], (X_pca[i, 0], X_pca[i, 1]))
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(FIGURE_DIR, 'pca_visualization.png'))
    plt.close()
    
    return X_pca, ipca.explained_variance_ratio_

def main():
    """Perform comprehensive analysis of protein fingerprints."""
    print("Loading fingerprint data...")
    df, probe_columns = load_fingerprint_data()
    print(f"Loaded {len(df)} proteins with {len(probe_columns)} probes")
    
    print("\nAnalyzing probe importance...")
    variances, correlations = analyze_probe_importance(df, probe_columns)
    print("Top 5 most distinctive probes:")
    print(variances.nlargest(5))
    
    # Try different clustering methods
    print("\nPerforming clustering analysis...")
    
    # 1. MiniBatch K-means
    clusters_kmeans, centers = cluster_proteins_minibatch(df, probe_columns, n_clusters=50)
    df['cluster_kmeans'] = clusters_kmeans
    print("\nMiniBatch K-means clustering completed")
    print("Cluster sizes:")
    print(pd.Series(clusters_kmeans).value_counts().head())
    
    # 2. HDBSCAN
    clusters_hdbscan = cluster_proteins_hdbscan(df, probe_columns, min_cluster_size=50)
    df['cluster_hdbscan'] = clusters_hdbscan
    print("\nHDBSCAN clustering completed")
    print("Number of clusters:", len(set(clusters_hdbscan)) - (1 if -1 in clusters_hdbscan else 0))
    
    # 3. BIRCH
    clusters_birch = cluster_proteins_birch(df, probe_columns, n_clusters=50)
    df['cluster_birch'] = clusters_birch
    print("\nBIRCH clustering completed")
    print("Cluster sizes:")
    print(pd.Series(clusters_birch).value_counts().head())
    
    print("\nFinding similar protein pairs...")
    similar_proteins = find_similar_proteins_efficient(df, probe_columns)
    print("Top 5 most similar protein pairs:")
    print(similar_proteins.head())
    
    print("\nPerforming PCA visualization...")
    X_pca, explained_var = visualize_pca_incremental(df, probe_columns)
    print(f"Total variance explained by first 2 PCs: {sum(explained_var):.2%}")
    
    # Save clustering results
    clustering_results = df[['pname', 'cluster_kmeans', 'cluster_hdbscan', 'cluster_birch']]
    clustering_results.to_csv(os.path.join(FIGURE_DIR, 'clustering_results.csv'), index=False)

if __name__ == "__main__":
    main()

# %%