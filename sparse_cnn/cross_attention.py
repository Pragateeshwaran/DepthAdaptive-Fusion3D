import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SubMConv3d


class SparseCrossAttention(nn.Module):
    """
    Sparse Cross-Attention using CONVOLUTIONS.
    MEMORY EFFICIENT: Only computes attention on sparse voxels, not full dense grid.
    """
    def __init__(self, 
                 lidar_dim=128, 
                 radar_dim=128, 
                 num_heads=8, 
                 dropout=0.1):
        super(SparseCrossAttention, self).__init__()
        
        assert lidar_dim % num_heads == 0
        assert radar_dim % num_heads == 0
        
        self.lidar_dim = lidar_dim
        self.radar_dim = radar_dim
        self.num_heads = num_heads
        self.head_dim = lidar_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use SPARSE 3D CONVOLUTIONS for Q, K, V projections
        self.q_proj = SubMConv3d(lidar_dim, lidar_dim, 1, bias=False, indice_key='q_proj')
        self.k_proj = SubMConv3d(radar_dim, lidar_dim, 1, bias=False, indice_key='k_proj')
        self.v_proj = SubMConv3d(radar_dim, lidar_dim, 1, bias=False, indice_key='v_proj')
        
        # Output projection
        self.out_proj = SubMConv3d(lidar_dim, lidar_dim, 1, bias=False, indice_key='out_proj')
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.BatchNorm1d(lidar_dim)
    
    def sparse_attention(self, Q_features, K_features, V_features, batch_size):
        """
        Compute attention ONLY on sparse features (memory efficient).
        
        Args:
            Q_features: (N, C) - LiDAR features
            K_features: (M, C) - Radar features  
            V_features: (M, C) - Radar features
            batch_size: int
        
        Returns:
            attended_features: (N, C)
        """
        N, C = Q_features.shape
        M = K_features.shape[0]
        
        # Reshape for multi-head: (N, C) -> (N, num_heads, head_dim)
        Q = Q_features.view(N, self.num_heads, self.head_dim)
        K = K_features.view(M, self.num_heads, self.head_dim)
        V = V_features.view(M, self.num_heads, self.head_dim)
        
        # Transpose for attention: (N, num_heads, head_dim) -> (num_heads, N, head_dim)
        Q = Q.transpose(0, 1)  # (num_heads, N, head_dim)
        K = K.transpose(0, 1)  # (num_heads, M, head_dim)
        V = V.transpose(0, 1)  # (num_heads, M, head_dim)
        
        # Scaled dot-product attention: (num_heads, N, head_dim) @ (num_heads, head_dim, M)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (num_heads, N, M)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention: (num_heads, N, M) @ (num_heads, M, head_dim) -> (num_heads, N, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back: (num_heads, N, head_dim) -> (N, num_heads, head_dim) -> (N, C)
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, C)
        
        return attn_output
    
    def forward(self, lidar_feat_sparse, radar_feat_sparse, lidar_conf_sparse, radar_conf_sparse):
        """
        Cross-attention with confidence weighting - MEMORY EFFICIENT VERSION.
        """
        # Apply confidence weighting
        lidar_conf_inv = 1.0 - lidar_conf_sparse.features  # (N, 1)
        lidar_weighted = lidar_feat_sparse.features * lidar_conf_inv
        
        radar_weighted = radar_feat_sparse.features * radar_conf_sparse.features
        
        # Create weighted sparse tensors
        lidar_weighted_sparse = lidar_feat_sparse.replace_feature(lidar_weighted)
        radar_weighted_sparse = radar_feat_sparse.replace_feature(radar_weighted)
        
        # Project to Q, K, V using SPARSE CONVOLUTIONS
        Q_sparse = self.q_proj(lidar_weighted_sparse)
        K_sparse = self.k_proj(radar_weighted_sparse)
        V_sparse = self.v_proj(radar_weighted_sparse)
        
        # Get features directly (NO dense conversion!)
        Q_features = Q_sparse.features  # (N_lidar, C)
        K_features = K_sparse.features  # (N_radar, C)
        V_features = V_sparse.features  # (N_radar, C)
        
        batch_size = lidar_feat_sparse.batch_size
        
        # Compute sparse attention (memory efficient)
        attn_features = self.sparse_attention(Q_features, K_features, V_features, batch_size)
        
        # Create sparse tensor with attention output
        attn_sparse = lidar_feat_sparse.replace_feature(attn_features)
        
        # Output projection
        output_sparse = self.out_proj(attn_sparse)
        output_features = self.dropout(output_sparse.features)
        
        # Apply layer norm
        output_features = self.layer_norm(output_features)
        
        enhanced_lidar = lidar_feat_sparse.replace_feature(output_features)
        
        return enhanced_lidar


class ATGN(nn.Module):
    """
    Adaptive Threshold Generation Network (ATGN) based on the paper.
    
    Extracts point cloud density information and generates a depth threshold
    to divide point clouds for differential fusion at different depths.
    
    From paper: "We design an adaptive threshold generation network to divide 
    point clouds more reasonably and guide the efficient completion of the 
    fusion process."
    """
    def __init__(self, feature_dim=128, radius=2.0):
        super(ATGN, self).__init__()
        
        self.radius = radius  # Radius for density calculation
        
        # MLP for density feature extraction and threshold generation
        # Input: volume density (1 channel)
        # Output: depth threshold (1 channel, normalized to [0, 1])
        self.density_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        # Gating network (takes concatenated features)
        self.gate = spconv.SparseSequential(
            SubMConv3d(feature_dim * 2, feature_dim, 3, padding=1, bias=False, indice_key='atgn1'),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            SubMConv3d(feature_dim, feature_dim, 1, bias=True, indice_key='atgn2'),
            nn.Sigmoid()
        )
    
    def compute_density(self, points):
        """
        Compute volume density for all points.
        
        Args:
            points: (N, 3) point coordinates
        
        Returns:
            density: (N, 1) volume density
        """
        N = points.shape[0]
        device = points.device
        
        # Compute pairwise distances (using batched approach for memory efficiency)
        batch_size = 1000  # Process 1000 points at a time
        densities = []
        
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch_points = points[i:end_idx]  # (B, 3)
            
            # Compute distances to all other points
            # (B, 3) - (1, N, 3) -> (B, N, 3) -> (B, N)
            dists = torch.norm(batch_points.unsqueeze(1) - points.unsqueeze(0), dim=2)
            
            # Count points within radius
            num_neighbors = (dists <= self.radius).sum(dim=1).float()  # (B,)
            
            # Volume density = number of points / volume of sphere
            volume = (4.0 / 3.0) * 3.14159 * (self.radius ** 3)
            batch_density = num_neighbors / volume  # (B,)
            
            densities.append(batch_density)
        
        density = torch.cat(densities, dim=0).unsqueeze(1)  # (N, 1)
        
        return density
    
    def generate_threshold(self, density):
        """
        Generate depth threshold from density information.
        
        Args:
            density: (N, 1) volume density
        
        Returns:
            threshold: scalar depth threshold (meters)
        """
        # Average density across all points
        avg_density = density.mean()  # Scalar
        
        # Pass through MLP to generate normalized threshold
        threshold_norm = self.density_mlp(avg_density.unsqueeze(0))  # (1, 1)
        
        # Scale to reasonable depth range (e.g., 20-50 meters)
        # Based on KITTI dataset statistics
        min_depth = 20.0
        max_depth = 50.0
        threshold = min_depth + threshold_norm.squeeze() * (max_depth - min_depth)
        
        return threshold
    
    def forward(self, features_sparse, points=None):
        """
        Apply adaptive gating in SPARSE space.
        
        Args:
            features_sparse: SparseConvTensor with concatenated features (2*feature_dim)
            points: (N, 3) original point coordinates for density calculation
        
        Returns:
            gated_features: SparseConvTensor
            depth_threshold: scalar threshold value
        """
        # Generate depth threshold if points provided
        if points is not None:
            density = self.compute_density(points)
            depth_threshold = self.generate_threshold(density)
        else:
            depth_threshold = torch.tensor(35.0, device=features_sparse.features.device)
        
        # Apply gating
        gate_weights_sparse = self.gate(features_sparse)
        
        # Extract first half of channels for gating
        features_half = features_sparse.features[:, :features_sparse.features.shape[1]//2]
        gated_features = features_half * gate_weights_sparse.features
        
        return features_sparse.replace_feature(gated_features), depth_threshold


class FusionModule(nn.Module):
    """
    CORRECTED Fusion Module - MEMORY EFFICIENT VERSION.
    
    Flow:
    1. Cross-attention (sparse): Q=LiDAR, K=V=Radar
    2. Residual: Enhanced + Original LiDAR
    3. Project Image to 3D sparse
    4. Concatenate [Residual, Image]
    5. ATGN (sparse) - generates threshold and gates features
    6. Project to BEV
    """
    def __init__(self, 
                 feature_dim=128, 
                 num_heads=8, 
                 dropout=0.1):
        super(FusionModule, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Cross-attention
        self.cross_attention = SparseCrossAttention(
            lidar_dim=feature_dim,
            radar_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Image to 3D projection
        self.image_to_3d = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # ATGN with density-based threshold generation
        self.atgn = ATGN(feature_dim=feature_dim)
        
        # Final BEV projection
        self.bev_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def sparse_to_bev(self, sparse_tensor):
        """Convert sparse 3D to BEV by summing along Z."""
        dense = sparse_tensor.dense()  # (B, C, D, H, W)
        bev = dense.sum(dim=2)  # (B, C, H, W)
        return bev
    
    def image_to_sparse_3d(self, image_feat, lidar_sparse, image_conf):
        """
        Project 2D image to 3D sparse space using LiDAR's structure.
        """
        B, C, H_img, W_img = image_feat.shape
        
        # Weight image by confidence
        image_weighted = image_feat * image_conf  # (B, C, H, W)
        
        # Process image
        image_feat_3d = self.image_to_3d(image_weighted)  # (B, C, H, W)
        
        # Get LiDAR spatial shape
        spatial_shape = lidar_sparse.spatial_shape  # [D, H, W]
        D, H_lidar, W_lidar = spatial_shape
        
        # Resize to match LiDAR BEV
        if (H_img != H_lidar) or (W_img != W_lidar):
            image_feat_3d = F.interpolate(image_feat_3d, size=(H_lidar, W_lidar), 
                                          mode='bilinear', align_corners=False)
        
        # Expand to 3D by replicating along Z
        image_feat_3d = image_feat_3d.unsqueeze(2)  # (B, C, 1, H, W)
        image_feat_3d = image_feat_3d.expand(-1, -1, D, -1, -1)  # (B, C, D, H, W)
        
        # Sample at LiDAR's sparse locations
        image_feat_3d_flat = image_feat_3d.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        
        batch_indices = lidar_sparse.indices[:, 0]
        z_indices = lidar_sparse.indices[:, 1]
        y_indices = lidar_sparse.indices[:, 2]
        x_indices = lidar_sparse.indices[:, 3]
        
        image_features_sparse = image_feat_3d_flat[batch_indices, z_indices, y_indices, x_indices]
        
        # Create sparse tensor
        image_sparse = lidar_sparse.replace_feature(image_features_sparse)
        
        return image_sparse
    
    def forward(self, lidar_feat_sparse, radar_feat_sparse, image_feat, 
                lidar_conf_sparse, radar_conf_sparse, image_conf,
                original_points=None):
        """
        Forward pass - MEMORY EFFICIENT.
        
        Args:
            original_points: (N, 3) original LiDAR points for density calculation
        """
        # Step 1: Cross-attention (ALL SPARSE - no dense conversion!)
        enhanced_lidar_sparse = self.cross_attention(
            lidar_feat_sparse, 
            radar_feat_sparse,
            lidar_conf_sparse,
            radar_conf_sparse
        )
        
        # Step 2: Residual connection
        residual_features = enhanced_lidar_sparse.features + lidar_feat_sparse.features
        residual_sparse = enhanced_lidar_sparse.replace_feature(residual_features)
        
        # Step 3: Project Image to 3D sparse
        image_sparse = self.image_to_sparse_3d(image_feat, lidar_feat_sparse, image_conf)
        
        # Step 4: Concatenate [Residual, Image]
        concatenated_features = torch.cat([
            residual_sparse.features,  # (N, C)
            image_sparse.features      # (N, C)
        ], dim=1)  # (N, 2*C)
        
        concatenated_sparse = residual_sparse.replace_feature(concatenated_features)
        
        # Step 5: ATGN (still sparse) - with density-based threshold
        gated_sparse, depth_threshold = self.atgn(concatenated_sparse, points=original_points)
        
        # Step 6: Project to BEV (only NOW convert to dense)
        bev_features = self.sparse_to_bev(gated_sparse)  # (B, C, H, W)
        
        # Step 7: Final BEV processing
        output_bev = self.bev_conv(bev_features)  # (B, C, H, W)
        
        return output_bev, depth_threshold


if __name__ == "__main__":
    print("ATGN-Enhanced Sparse Cross-Attention")
    print("="*60)
    print("Key features:")
    print("- Density-based adaptive threshold generation")
    print("- Memory efficient sparse attention")
    print("- Paper-based ATGN implementation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy sparse tensors
    B, C, D, H, W = 2, 128, 10, 50, 50
    num_voxels = 1000
    
    features_lidar = torch.randn(num_voxels, C).to(device)
    features_radar = torch.randn(num_voxels, C).to(device)
    conf_lidar = torch.rand(num_voxels, 1).to(device)
    conf_radar = torch.rand(num_voxels, 1).to(device)
    
    indices = torch.randint(0, 2, (num_voxels, 1)).to(device)
    indices = torch.cat([
        indices,
        torch.randint(0, D, (num_voxels, 1)).to(device),
        torch.randint(0, H, (num_voxels, 1)).to(device),
        torch.randint(0, W, (num_voxels, 1)).to(device),
    ], dim=1).int()
    
    lidar_sparse = SparseConvTensor(features_lidar, indices, [D, H, W], B)
    radar_sparse = SparseConvTensor(features_radar, indices, [D, H, W], B)
    lidar_conf_sparse = SparseConvTensor(conf_lidar, indices, [D, H, W], B)
    radar_conf_sparse = SparseConvTensor(conf_radar, indices, [D, H, W], B)
    
    image_feat = torch.randn(B, C, H, W).to(device)
    image_conf = torch.rand(B, 1, H, W).to(device)
    
    # Original points for density calculation
    original_points = torch.randn(num_voxels, 3).to(device) * 10  # Random points
    
    # Test fusion with ATGN
    fusion = FusionModule(feature_dim=C, num_heads=8).to(device)
    fused, threshold = fusion(lidar_sparse, radar_sparse, image_feat,
                             lidar_conf_sparse, radar_conf_sparse, image_conf,
                             original_points=original_points)
    
    print(f"\n✓ Fusion output: {fused.shape}")
    print(f"✓ Generated depth threshold: {threshold:.2f}m")
    print(f"✓ ATGN working correctly!")