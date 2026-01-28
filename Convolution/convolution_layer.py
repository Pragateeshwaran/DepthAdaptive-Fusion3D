import torch
import torch.nn as nn
import torch.nn.functional as F
"""
image input = (375, 1200, 3)
hidden state = (128, 375, 1200)
output = (375, 1200, 3)
"""
class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=3, padding=1):
        super(ConvolutionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channels, in_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        # out: (B, C, H, W) -> (B, H, W, C)
        out = out.permute(0, 2, 3, 1)
        return out
    
