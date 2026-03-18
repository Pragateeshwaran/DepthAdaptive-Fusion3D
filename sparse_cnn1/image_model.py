import torch
import torch.nn as nn


class ImageFeatureExtractor(nn.Module):
    """
    Image Feature Extractor with dual branches (features + confidence).
    Both branches use 2D Convolutions.
    """
    def __init__(self, in_channels=3, feature_dim=128):
        super(ImageFeatureExtractor, self).__init__()
        
        # ============ Shared Encoder (Dimension Reduction) ============
        self.encoder = nn.Sequential(
            # Input: 3 -> 64
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Additional refinement
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # ============ Feature Branch (Dimension Increase then Reduce) ============
        self.feature_branch = nn.Sequential(
            # 512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256 -> 512 (increase)
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256 -> feature_dim
            nn.Conv2d(256, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # ============ Confidence Branch (Dimension Reduction to 1) ============
        self.confidence_branch = nn.Sequential(
            # 512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256 -> 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128 -> 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64 -> 1 (confidence score)
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  # [0, 1] confidence
        )
    
    def forward(self, images):
        """
        Forward pass through dual branches.
        
        Args:
            images (torch.Tensor): Input images (B, C, H, W)
        
        Returns:
            features (torch.Tensor): Feature tensor (B, feature_dim, H', W')
            confidence (torch.Tensor): Confidence tensor (B, 1, H', W')
        """
        # Shared encoder
        encoded = self.encoder(images)
        
        # Dual branches
        features = self.feature_branch(encoded)
        confidence = self.confidence_branch(encoded)
        
        return features, confidence


if __name__ == "__main__":
    print("Image Feature Extractor - Standalone Test")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageFeatureExtractor(in_channels=3, feature_dim=128).to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 375, 1242).to(device)
    features, confidence = model(dummy_input)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Confidence shape: {confidence.shape}")