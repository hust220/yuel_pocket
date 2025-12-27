import numpy as np
from scipy.spatial import KDTree

def hill_climbing_cluster(points, scores, k_nn=10, min_size=3, max_clusters=100):
    """
    Cluster points using Hill-Climbing to local peaks and Union-Find grouping.
    
    Args:
        points: (N, 3) numpy array of coordinates
        scores: (N,) numpy array of scores
        k_nn: Number of nearest neighbors to consider for hill climbing
        min_size: Minimum number of points in a cluster
        max_clusters: Keep top N clusters
        
    Returns:
        labels: (N,) array with cluster ID for each point. -1 means noise/unclustered.
        clusters: List of dictionaries containing cluster info
    """
    if len(points) == 0:
        return np.array([], dtype=int), []
        
    tree = KDTree(points)
    k_actual = min(k_nn + 1, len(points))
    _, idx = tree.query(points, k=k_actual)
    
    parent = np.arange(len(points))
    values = scores
    
    for i in range(len(points)):
        # Neighbors excluding self (index 0 is usually self in KDTree query)
        neigh = idx[i][1:]
        neigh = neigh[neigh < len(points)]
        
        if len(neigh) > 0:
            # Find neighbor with highest score
            j = neigh[np.argmax(values[neigh])]
            if values[j] > values[i]:
                parent[i] = j

    def find(i, parent_arr):
        root = i
        while parent_arr[root] != root:
            root = parent_arr[root]
        # Path compression
        curr = i
        while parent_arr[curr] != root:
            next_p = parent_arr[curr]
            parent_arr[curr] = root
            curr = next_p
        return root

    peaks = {}
    for i in range(len(points)):
        r = find(i, parent)
        peaks.setdefault(r, []).append(i)
        
    labels = np.full(len(points), -1, dtype=int)
    clusters = []
    
    # Pre-sort peaks by score to assign IDs in order of importance
    sorted_peak_indices = sorted(peaks.keys(), key=lambda x: scores[x], reverse=True)
    
    final_id = 0
    for r in sorted_peak_indices:
        members = peaks[r]
        if len(members) >= min_size:
            labels[members] = final_id
            clusters.append({
                'id': final_id,
                'score': float(scores[r]),
                'center': points[r],
                'size': len(members),
                'indices': np.array(members)
            })
            final_id += 1
            if final_id >= max_clusters:
                break
                
    return labels, clusters
