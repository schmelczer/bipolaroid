import torch.nn as nn


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

    def forward(self, x):
        return self.conv(x) + x


# Define the network
class Res2(nn.Module):
    def __init__(self, bin_count):
        super(Res2, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(16), ResidualBlock(16), ResidualBlock(16), ResidualBlock(16)
        )
        self.output_layer = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x
