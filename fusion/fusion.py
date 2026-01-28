import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.kitti_loader as kitti_loader
from GNN.PointGNN import PointGNNLayer
from Convolution.convolution_layer import ConvolutionLayer
from calib import calib_lidar_to_camera
class FusionLayer(nn.Module):
 
    def __init__(self, num_iterations=3, state_dim=128, num_classes=4):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Create T iterations of GNN layers (each with different parameters)
        self.Glayers = nn.ModuleList([
            PointGNNLayer(state_dim) for _ in range(num_iterations)
        ])

        self.Clayers = nn.ModuleList([
            ConvolutionLayer(in_channels=state_dim, out_channels=state_dim) for _ in range(num_iterations)
        ])
        
        self.alpha_mlp = nn.Sequential(
            nn.Linear(3, 32),   # LiDAR xyz
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

       

    def forward(self, graph_data, img_data, calib_data, lidar_points_prcoessed):
        """
        Args:
            graph_data: dict with keys:
                - 'vertex_features': initial vertex states (N, state_dim)
                - 'vertex_coords': vertex 3D coordinates (N, 3)
                - 'edge_index': graph connectivity (2, E)
        Returns:
            cls_logits: classification logits (N, num_classes)
            bbox_pred: bounding box predictions (N, 7)
        """
        s = graph_data['vertex_features']  # (N, state_dim)
        coords = graph_data['vertex_coords']  # (N, 3)
        edge_index = graph_data['edge_index']  # (2, E)
        
        # Iterate through GNN layers
        for i in range(self.num_iterations):
            s_gnn = self.Glayers[i](s, coords, edge_index)
            s_conv = self.Clayers[i](img_data)
            calib_lidar = calib_lidar_to_camera(lidar_points_prcoessed, calib_data)
            # step 1 ==========================================================================
            interpolation = F.grid_sample(
                s_conv.permute(0, 3, 1, 2),  # (B, C, H, W)
                calib_lidar.unsqueeze(0).unsqueeze(0).float(),  # (1, 1, N, 2)
                align_corners=True
            ).squeeze().permute(1, 0)  # (N, C)
            # Compute alpha from LiDAR geometry
            alpha = self.alpha_mlp(coords)  # (N, 1)

            # Expand alpha to match feature dimension
            alpha = alpha.expand_as(s_gnn)  # (N, state_dim)

            # Gated fusion
            s = alpha * s_gnn + (1.0 - alpha) * interpolation

            # step 2 ==========================================================================
            # r = torch.norm(coords, dim=1, keepdim=True)  # (N, 1)
            # s = r * s_gnn + (1.0 - r) * interpolation

            

        # Predict classification and bounding box
        cls_logits = self.MLP_cls(s)  # (N, num_classes)
        bbox_pred = self.MLP_loc(s)  # (N, 7)
        
        return cls_logits, bbox_pred
 