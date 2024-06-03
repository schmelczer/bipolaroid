import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class PhotoEnhanceNetAdvanced(nn.Module):
    def __init__(self, bin_count):
        super(PhotoEnhanceNetAdvanced, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16), ConvBlock(16, 32), ConvBlock(32, 64), ConvBlock(64, 128)
        )
        self.channel_attention = ChannelAttention(128)
        self.final_conv = nn.Conv3d(128, 1, kernel_size=1)  # Reduce channel size to 1

    def forward(self, x):
        x = self.features(x)
        x = self.channel_attention(x)
        x = self.final_conv(x)  # Final reduction to match the input channel dimensions
        return x
