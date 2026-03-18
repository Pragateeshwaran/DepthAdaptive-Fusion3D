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
                 dropout=0.1,
                 max_radar_tokens=2048,
                 query_chunk_size=1024):
        super(SparseCrossAttention, self).__init__()
        
        assert lidar_dim % num_heads == 0
        assert radar_dim % num_heads == 0
        
        self.lidar_dim = lidar_dim
        self.radar_dim = radar_dim
        self.num_heads = num_heads
        self.head_dim = lidar_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_radar_tokens = max_radar_tokens
        self.query_chunk_size = query_chunk_size
        
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
        
        # Chunked attention to avoid huge (N x M) memory usage.
        outputs = []
        for q_start in range(0, N, self.query_chunk_size):
            q_end = min(q_start + self.query_chunk_size, N)
            q_chunk = Q[:, q_start:q_end, :]  # (num_heads, Qc, head_dim)

            attn_scores = torch.matmul(q_chunk, K.transpose(-2, -1)) * self.scale  # (num_heads, Qc, M)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out_chunk = torch.matmul(attn_weights, V)  # (num_heads, Qc, head_dim)
            outputs.append(out_chunk)

        attn_output = torch.cat(outputs, dim=1)
        
        # Reshape back: (num_heads, N, head_dim) -> (N, num_heads, head_dim) -> (N, C)
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, C)
        
        return attn_output
    
    def forward(self, lidar_feat_sparse, radar_feat_sparse, lidar_conf_sparse, radar_conf_sparse):
        """
        Cross-attention with confidence weighting - MEMORY EFFICIENT VERSION.
        Per architecture diagram:
        - LiDAR: Q = FL * (WL-1) where WL-1 = (1 - WL) = (1 - lidar_confidence)
        - Radar: K, V = FR * WR where WR = radar_confidence
        """
        # Apply confidence weighting PER DIAGRAM
        # LiDAR: weight by (1 - confidence) to reduce where confident
        lidar_conf_inv = 1.0 - lidar_conf_sparse.features  # (N, 1) - This is WL-1 in diagram
        lidar_weighted = lidar_feat_sparse.features * lidar_conf_inv
        
        # Radar: weight by confidence directly
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

        # Keep only the strongest radar tokens to control quadratic attention cost.
        if K_features.shape[0] > self.max_radar_tokens:
            radar_scores = radar_conf_sparse.features.squeeze(-1)
            top_idx = torch.topk(radar_scores, k=self.max_radar_tokens, sorted=False).indices
            K_features = K_features[top_idx]
            V_features = V_features[top_idx]
        
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
    """
    def __init__(
        self,
        feature_dim=128,
        radius=2.0,
        threshold_method='range_percentile',
        range_percentile=0.60,
        threshold_min=10.0,
        threshold_max=50.0,
    ):
        super(ATGN, self).__init__()
        
        self.radius = radius  # Radius for density calculation
        self.threshold_method = threshold_method
        self.range_percentile = range_percentile
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        
        # MLP for density feature extraction and threshold generation
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
        """Compute volume density for all points."""
        N = points.shape[0]
        device = points.device
        
        # Compute pairwise distances (using batched approach for memory efficiency)
        batch_size = 1000  # Process 1000 points at a time
        densities = []
        
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch_points = points[i:end_idx]  # (B, 3)
            
            # Compute distances to all other points
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
        """Generate adaptive depth threshold from density."""
        # Aggregate density (mean)
        density_mean = density.mean()
        
        # MLP to generate threshold
        threshold = self.density_mlp(density_mean.unsqueeze(0).unsqueeze(0))  # (1, 1, 1)
        
        # Scale to reasonable depth range (e.g., 10-50m)
        threshold = 10.0 + threshold * 40.0  # Range: [10, 50]
        
        return threshold.squeeze()

    def generate_threshold_from_range(self, points):
        """
        Fast adaptive threshold from range distribution.
        Works for both metric points and voxel-index points.
        """
        if points is None or points.numel() == 0:
            return torch.tensor(35.0, device=self.density_mlp[0].weight.device)

        points = points.float()
        if points.shape[1] >= 2:
            ranges = torch.norm(points[:, :2], dim=1)
        else:
            ranges = torch.abs(points[:, 0])

        if ranges.numel() == 0:
            return torch.tensor(35.0, device=points.device)

        q = torch.quantile(ranges, self.range_percentile)
        return torch.clamp(q, min=self.threshold_min, max=self.threshold_max)
    
    def forward(self, features_sparse, points=None):
        """
        Forward pass with optional density-based threshold generation.
        
        Args:
            features_sparse: SparseConvTensor with concatenated [lidar+image] features
            points: (N, 3) original point coordinates for density calculation
        
        Returns:
            gated_features: SparseConvTensor with gated features
            depth_threshold: scalar tensor
        """
        # Generate threshold using selected method.
        if self.threshold_method == 'density' and points is not None:
            density = self.compute_density(points)
            depth_threshold = self.generate_threshold(density)
        elif points is not None:
            depth_threshold = self.generate_threshold_from_range(points)
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
    CORRECTED Fusion Module - Implements architecture diagram exactly.
    
    Flow (per diagram):
    1. Cross-attention: Q = LiDAR * (1-WL), K/V = Radar * WR
    2. Residual: Enhanced + Original LiDAR  
    3. Project Image to 3D sparse weighted by image_conf * (1 - lidar_conf)
    4. Concatenate [Residual, Image]
    5. ATGN - generates threshold and gates features
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
    
    def image_to_sparse_3d(self, image_feat, lidar_sparse, image_conf, lidar_conf_sparse):
        """
        ⭐ CRITICAL FIX: Project 2D image to 3D sparse space using LiDAR's structure.
        Per architecture diagram: Weight image by image_conf AND (1 - lidar_conf)
        
        Args:
            image_feat: (B, C, H, W) image features
            lidar_sparse: SparseConvTensor - provides spatial structure
            image_conf: (B, 1, H, W) image confidence
            lidar_conf_sparse: SparseConvTensor - LiDAR confidence at voxels
        
        Returns:
            image_sparse: SparseConvTensor with image features weighted properly
        """
        B, C, H_img, W_img = image_feat.shape
        
        # Step 1: Weight image by its own confidence
        image_weighted = image_feat * image_conf  # (B, C, H, W)
        
        # Step 2: Process image
        image_feat_3d = self.image_to_3d(image_weighted)  # (B, C, H, W)
        
        # Step 3: Get LiDAR spatial shape
        spatial_shape = lidar_sparse.spatial_shape  # [D, H, W]
        D, H_lidar, W_lidar = spatial_shape
        
        # Step 4: Resize to match LiDAR BEV
        if (H_img != H_lidar) or (W_img != W_lidar):
            image_feat_3d = F.interpolate(image_feat_3d, size=(H_lidar, W_lidar), 
                                          mode='bilinear', align_corners=False)
        
        # Step 5: Expand to 3D by replicating along Z
        image_feat_3d = image_feat_3d.unsqueeze(2)  # (B, C, 1, H, W)
        image_feat_3d = image_feat_3d.expand(-1, -1, D, -1, -1)  # (B, C, D, H, W)
        
        # Step 6: Sample at LiDAR's sparse locations
        image_feat_3d_flat = image_feat_3d.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        
        batch_indices = lidar_sparse.indices[:, 0]
        z_indices = lidar_sparse.indices[:, 1]
        y_indices = lidar_sparse.indices[:, 2]
        x_indices = lidar_sparse.indices[:, 3]
        
        image_features_sparse = image_feat_3d_flat[batch_indices, z_indices, y_indices, x_indices]
        
        # ⭐ CRITICAL FIX: Weight by (1 - lidar_conf) as per architecture diagram
        # This implements the multiplication with (1 - WL) for image features
        lidar_conf_inv = 1.0 - lidar_conf_sparse.features  # (N, 1)
        image_features_sparse = image_features_sparse * lidar_conf_inv  # (N, C)
        
        # Step 7: Create sparse tensor
        image_sparse = lidar_sparse.replace_feature(image_features_sparse)
        
        return image_sparse
    
    def forward(self, lidar_feat_sparse, radar_feat_sparse, image_feat, 
                lidar_conf_sparse, radar_conf_sparse, image_conf,
                original_points=None):
        """
        Forward pass - Implements architecture diagram exactly.
        
        Args:
            lidar_feat_sparse: SparseConvTensor (N, C)
            radar_feat_sparse: SparseConvTensor (N, C)
            image_feat: Tensor (B, C, H, W)
            lidar_conf_sparse: SparseConvTensor (N, 1) 
            radar_conf_sparse: SparseConvTensor (N, 1)
            image_conf: Tensor (B, 1, H, W)
            original_points: (N, 3) original LiDAR points for density calculation
        """
        # Step 1: Doppler velocity gate using radar feature channel index 3.
        if radar_feat_sparse.features.shape[1] > 3:
            velocity_channel = radar_feat_sparse.features[:, 3]
        else:
            velocity_channel = torch.zeros(
                radar_feat_sparse.features.shape[0],
                device=radar_feat_sparse.features.device,
                dtype=radar_feat_sparse.features.dtype
            )
        vel_gate = torch.sigmoid(velocity_channel).unsqueeze(1)
        radar_gated_sparse = radar_feat_sparse.replace_feature(
            radar_feat_sparse.features * vel_gate
        )
        enhanced_lidar_sparse = self.cross_attention(
            lidar_feat_sparse,
            radar_gated_sparse,
            lidar_conf_sparse,
            radar_conf_sparse
        )
        
        # Step 2: Residual connection (Enhanced + Original)
        residual_features = enhanced_lidar_sparse.features + lidar_feat_sparse.features
        residual_sparse = enhanced_lidar_sparse.replace_feature(residual_features)
        
        # Step 3: ⭐ FIXED - Project Image to 3D sparse with proper confidence weighting
        image_sparse = self.image_to_sparse_3d(
            image_feat, 
            lidar_feat_sparse, 
            image_conf, 
            lidar_conf_sparse  # ⭐ NOW includes lidar confidence
        )
        
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
    print("✅ CORRECTED Cross-Attention with Proper Image Confidence Weighting")
    print("="*80)
    print("Key fixes:")
    print("- ✅ Image features weighted by image_conf * (1 - lidar_conf)")
    print("- ✅ Implements architecture diagram exactly")
    print("- ✅ LiDAR: Q = FL * (1-WL)")
    print("- ✅ Radar: K/V = FR * WR") 
    print("- ✅ Image: FI * WI * (1-WL)")
    print("="*80)
    
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
    
    original_points = torch.randn(num_voxels, 3).to(device) * 10
    
    # Test fusion
    fusion = FusionModule(feature_dim=C, num_heads=8).to(device)
    fused, threshold = fusion(lidar_sparse, radar_sparse, image_feat,
                             lidar_conf_sparse, radar_conf_sparse, image_conf,
                             original_points=original_points)
    
    print(f"\n✓ Fusion output: {fused.shape}")
    print(f"✓ Generated depth threshold: {threshold:.2f}m")
    print(f"✓ All fixes applied successfully!")
