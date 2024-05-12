import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class SmartRes(nn.Module):
    def __init__(self, bin_count):
        super(SmartRes, self).__init__()
        self.bin_count = bin_count
        self.initial_conv = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm3d(16)
        self.resblock1 = ResidualBlock(16)
        self.resblock2 = ResidualBlock(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.dilated_conv = nn.Conv3d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.bn_dilated = nn.BatchNorm3d(32)
        self.final_conv = nn.Conv3d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn0(self.initial_conv(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn_dilated(self.dilated_conv(x)))
        x = self.final_conv(x)
        return x
