import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedAestheticHistogramNet(nn.Module):
    def __init__(self, bin_count):
        super(EnhancedAestheticHistogramNet, self).__init__()
        self.bin_count = bin_count

        # Initial convolution layer
        self.initial_conv = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm3d(32)
        self.initial_relu = nn.ReLU()

        # Deeper convolutional layers with increasing channels and dilation for expanded receptive field
        self.conv1 = self._make_layer(32, 64, dilation=1)
        self.conv2 = self._make_layer(64, 128, dilation=2)
        self.conv3 = self._make_layer(128, 256, dilation=4)

        # Attention module
        self.attention = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1),  # Pointwise convolution
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=1),  # Pointwise convolution
            nn.Sigmoid(),
        )

        # Correctly adjusted residual connections
        self.res1 = nn.Conv3d(
            1, 256, kernel_size=1
        )  # Match initial input channels to later layers
        self.res2 = nn.Conv3d(128, 256, kernel_size=1)  # Match output of conv2 to conv3

        # Final convolution to bring channels back to 1
        self.final_conv = nn.Conv3d(256, 1, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, in_channels, out_channels, dilation):
        layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        return layer

    def forward(self, x):
        identity1 = self.res1(x)  # First skip connection

        out = self.initial_relu(self.initial_bn(self.initial_conv(x)))
        out = self.conv1(out)
        out = self.conv2(out)

        identity2 = self.res2(out)  # Second skip connection

        out = self.conv3(out)
        out = self.attention(out) * out  # Apply attention

        out += identity2  # Add from second skip connection
        out += identity1  # Add from first skip connection

        out = self.final_conv(out)
        return out
