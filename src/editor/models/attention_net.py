import torch
import torch.nn as nn


# Define the self-attention module
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv3d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        query = (
            self.query_conv(x)
            .view(batch_size, -1, depth * height * width)
            .permute(0, 2, 1)
        )
        key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        attention = self.softmax(torch.bmm(query, key))  # Batch matrix multiplication
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)

        return x + out


# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
        )
        self.attention = SelfAttention(channels)

    def forward(self, x):
        return self.attention(self.conv(x)) + x


# Define the network
class AttentionNet(nn.Module):
    def __init__(self, bin_count):
        super(AttentionNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16),
        )
        self.res_blocks = nn.Sequential(ResidualBlock(16), ResidualBlock(16))
        self.output_layer = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x
