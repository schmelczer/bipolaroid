import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotoEnhanceNetAdvanced(nn.Module):
    def __init__(self, bin_count):
        super(PhotoEnhanceNetAdvanced, self).__init__()
        self.bin_count = bin_count

        # Enhance complexity of the network
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        # Adjusted attention layer to match features channel size
        self.attention = nn.Sequential(
            nn.Conv3d(
                128, 128, 1
            ),  # Ensure the attention map has the same number of channels
            nn.Sigmoid(),
        )

        # Using dense connections
        self.dense = nn.Sequential(
            nn.Conv3d(
                256, 192, kernel_size=3, padding=1
            ),  # Adjust input channels to account for concatenated layers
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.Conv3d(192, 1, kernel_size=1),
        )

    def forward(self, x):
        features = self.features(x)

        # Apply attention
        attention = self.attention(features)
        x = features * attention  # Element-wise multiplication

        # Concatenate for dense connection (skip connection)
        x = torch.cat((features, attention), dim=1)  # Combining feature maps

        x = self.dense(x)
        return x
