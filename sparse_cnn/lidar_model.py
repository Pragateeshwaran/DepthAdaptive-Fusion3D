import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SubMConv3d, SparseConv3d


class LiDARFeatureExtractor(nn.Module):
    """
    LiDAR Feature Extractor with dual branches (features + confidence).
    Both branches use Sparse 3D Convolutions.
    """
    def __init__(self, in_channels=3, feature_dim=128):
        super(LiDARFeatureExtractor, self).__init__()
        
        # ============ Shared Encoder (Dimension Reduction) ============
        self.encoder = spconv.SparseSequential(
            # Input: in_channels -> 16
            SubMConv3d(in_channels, 16, 3, padding=1, bias=False, indice_key='subm0'),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # 16 -> 32 (reduce spatial)
            SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            SubMConv3d(32, 32, 3, padding=1, bias=False, indice_key='subm1'),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # 32 -> 64 (reduce spatial)
            SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key='subm2'),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # 64 -> 128 (reduce spatial)
            SparseConv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            SubMConv3d(128, 128, 3, padding=1, bias=False, indice_key='subm3'),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # ============ Feature Branch (Dimension Increase) ============
        self.feature_branch = spconv.SparseSequential(
            # 128 -> 256
            SubMConv3d(128, 256, 3, padding=1, bias=False, indice_key='feat1'),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # 256 -> 128
            SubMConv3d(256, 128, 3, padding=1, bias=False, indice_key='feat2'),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # 128 -> feature_dim
            SubMConv3d(128, feature_dim, 1, bias=False, indice_key='feat3'),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # ============ Confidence Branch (Dimension Reduction to 1) ============
        self.confidence_branch = spconv.SparseSequential(
            # 128 -> 64
            SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='conf1'),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # 64 -> 32
            SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='conf2'),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # 32 -> 1 (confidence score)
            SubMConv3d(32, 1, 1, bias=True, indice_key='conf3'),
            nn.Sigmoid()  # [0, 1] confidence
        )
    
    def forward(self, sparse_tensor):
        """
        Forward pass through dual branches.
        
        Args:
            sparse_tensor (SparseConvTensor): Input sparse tensor
        
        Returns:
            features (SparseConvTensor): Feature tensor
            confidence (SparseConvTensor): Confidence tensor
        """
        # Shared encoder
        encoded = self.encoder(sparse_tensor)
        
        # Dual branches
        features = self.feature_branch(encoded)
        confidence = self.confidence_branch(encoded)
        
        return features, confidence


if __name__ == "__main__":
    print("LiDAR Feature Extractor - Standalone Test")
    print("Note: This model expects SparseConvTensor input")
    print("Voxelization should be done in the training loop")
    
    model = LiDARFeatureExtractor(in_channels=3, feature_dim=128)
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")