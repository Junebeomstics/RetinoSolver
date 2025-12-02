"""
Utility functions for model training
"""
import torch


def compute_edge_features(pos, edge_index, k=5):
    """
    Compute edge-based features for each node:
    - Average distance to k nearest neighbors
    - Node degree (number of connections)
    - Local density estimate
    
    Args:
        pos: Node positions (N, 3)
        edge_index: Edge indices (2, E)
        k: Number of nearest neighbors to consider
    
    Returns:
        edge_features: (N, 3) tensor containing [avg_knn_dist, node_degree, local_density]
    """
    num_nodes = pos.shape[0]
    device = pos.device
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(pos, pos)  # (N, N)
    
    # Get k nearest neighbors for each node
    _, k_nearest_indices = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)  # k+1 to exclude self
    k_nearest_indices = k_nearest_indices[:, 1:]  # Remove self
    
    # Average distance to k nearest neighbors
    k_nearest_dists = dist_matrix.gather(1, k_nearest_indices)  # (N, k)
    avg_knn_dist = k_nearest_dists.mean(dim=1, keepdim=True)  # (N, 1)
    
    # Node degree (number of edges)
    if edge_index is not None and edge_index.numel() > 0:
        node_degree = torch.zeros(num_nodes, 1, device=device)
        unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
        node_degree[unique_nodes] = counts.float().unsqueeze(1)
    else:
        node_degree = torch.zeros(num_nodes, 1, device=device)
    
    # Local density: inverse of average distance to neighbors
    local_density = 1.0 / (avg_knn_dist + 1e-6)  # (N, 1)
    
    # Combine edge features
    edge_features = torch.cat([avg_knn_dist, node_degree, local_density], dim=1)  # (N, 3)
    
    return edge_features

