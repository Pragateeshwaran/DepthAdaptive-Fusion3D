import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import torch
import torch.nn as nn
from typing import Optional
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import data_loader  

# Suppress PyG warnings about missing extensions
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')


# --------------------------- Vertex Initialization ---------------------------

class LocalFeatureMLP(nn.Module):
    """Encodes raw points around each vertex."""
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=128):
        super(LocalFeatureMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


def build_radius_graph_gpu(coords_tensor, radius, max_neighbors=128):
    """
    GPU-based radius graph construction using PyTorch (no external dependencies)
    
    Args:
        coords_tensor: (N, 3) tensor on GPU
        radius: connection radius
        max_neighbors: maximum neighbors per vertex
    Returns:
        edge_index: (2, E) edge list
    """
    N = coords_tensor.shape[0]
    device = coords_tensor.device
    
    # Compute pairwise distances (N x N matrix)
    # dist[i,j] = ||coords[i] - coords[j]||
    diff = coords_tensor.unsqueeze(0) - coords_tensor.unsqueeze(1)  # (N, N, 3)
    dist = torch.norm(diff, dim=2)  # (N, N)
    
    # Find neighbors within radius (excluding self-loops)
    mask = (dist < radius) & (dist > 0)  # (N, N)
    
    # Convert to edge list
    edge_index = mask.nonzero(as_tuple=False).t()  # (2, E)
    
    return edge_index


def build_radius_graph_cpu(coords, radius):
    """
    CPU-based radius graph construction using cKDTree
    
    Args:
        coords: (N, 3) numpy array
        radius: connection radius
    Returns:
        edge_index: (2, E) numpy array
    """
    tree = cKDTree(coords)
    edge_list = []
    
    for i in range(len(coords)):
        neighbors = tree.query_ball_point(coords[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        for j in neighbors:
            edge_list.append([i, j])
    
    if len(edge_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    else:
        return np.array(edge_list, dtype=np.int64).T


def construct_graph(points, radius=4.0, num_vertices=1000, device=None, verbose=True):
    """
    Construct graph from point cloud with GPU support.
    
    Args:
        points: (M, 3 or 4) input point cloud (x, y, z) or (x, y, z, intensity)
        radius: edge connection radius
        num_vertices: number of vertices to downsample to
        device: torch device (cuda or cpu)
        verbose: if True, print progress (only use for first batch)
    Returns:
        dict with 'vertex_features', 'vertex_coords', 'edge_index'
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Constructing graph from {len(points)} points on {device}...")
    
    # Step 1: Downsample
    downsampled_points = data_loader.lidar_fps_gpu(points, N=num_vertices)
    
    # Handle missing intensity column
    if downsampled_points.shape[1] == 3:
        # Add zeros for intensity if missing
        intensity = np.zeros((len(downsampled_points), 1))
        downsampled_points = np.hstack([downsampled_points, intensity])
    
    # Step 2: Build edges
    if device.type == 'cuda':
        # Use GPU-based graph construction
        coords_tensor = torch.tensor(downsampled_points[:, :3], dtype=torch.float32).to(device)
        edge_index = build_radius_graph_gpu(coords_tensor, radius)
        edge_index = edge_index.cpu().numpy()
    else:
        # Use CPU-based graph construction
        edge_index = build_radius_graph_cpu(downsampled_points[:, :3], radius)
    
    # Step 3: Initialize vertex features (batched on GPU)
    s0 = initialize_vertex_features_batched(downsampled_points, edge_index, device=device, verbose=verbose)
    
    # Convert to tensors and move to device
    s0 = torch.tensor(s0, dtype=torch.float32).to(device)
    vertex_pos = torch.tensor(downsampled_points[:, :3], dtype=torch.float32).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    
    if verbose:
        print(f"Graph: {len(vertex_pos)} vertices, {edge_index.shape[1]} edges")
        print(f"All tensors on device: {device}")
    
    return {
        'vertex_features': s0,
        'vertex_coords': vertex_pos,
        'edge_index': edge_index,
    }


def initialize_vertex_features_batched(downsampled_points, edge_index, device=None, verbose=True):
    """
    Initialize features with GPU support using BATCHED processing
    
    Args:
        downsampled_points: (N, 4) numpy array with x, y, z, intensity
        edge_index: (2, E) numpy array of edges
        device: torch device
        verbose: print progress
    Returns:
        vertex_features: (N, 128) numpy array
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = len(downsampled_points)
    
    # Create MLP and move to GPU
    mlp = LocalFeatureMLP(input_dim=4, hidden_dim=64, output_dim=128).to(device)
    mlp.eval()  # Set to eval mode (no training)
    
    # Convert all points to tensors on GPU once
    all_points_gpu = torch.tensor(downsampled_points, dtype=torch.float32).to(device)
    
    # Convert edge_index to tensor
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(device)
    
    vertex_features = []
    
    with torch.no_grad():  # No gradient computation needed
        for i in range(N):
            if verbose and i % 100 == 0:
                print(f"Processing vertex {i+1}/{N}", end='\r')
            
            # Find neighbors from edge_index
            mask = edge_index_tensor[0] == i
            neighbors_idx = edge_index_tensor[1][mask]
            
            if len(neighbors_idx) == 0:
                vertex_features.append(torch.zeros(128, device=device))
                continue
            
            # Get neighbor points (already on GPU)
            local_points = all_points_gpu[neighbors_idx]
            center_point = all_points_gpu[i]
            
            # Compute relative coordinates
            rel = local_points[:, :3] - center_point[:3]
            intensity = local_points[:, 3:4]
            
            # Concatenate features
            inputs = torch.cat([rel, intensity], dim=1)
            
            # Forward pass on GPU (batched for all neighbors at once)
            local_feats = mlp(inputs)
            
            # Max pooling
            pooled, _ = torch.max(local_feats, dim=0)
            
            vertex_features.append(pooled)
    
    if verbose:
        print()  # New line after progress
    
    # Stack and move to CPU only at the end
    vertex_features = torch.stack(vertex_features).cpu().numpy()
    
    return vertex_features


# --------------------------- Visualization ---------------------------

def visualize_graph(points, radius=4.0, num_vertices=1000):
    """
    Visualize the LiDAR graph using Open3D.
    """
    # Construct graph
    graph_data = construct_graph(points, radius=radius, num_vertices=num_vertices)
    
    points_np = graph_data['vertex_coords'].cpu().numpy()
    edge_index_np = graph_data['edge_index'].cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(edge_index_np.T.astype(np.int32))
    
    o3d.visualization.draw_geometries([pcd, line_set])