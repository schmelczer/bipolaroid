import torch.nn as nn
import torch.nn.functional as F
import torch


class HistogramRestorationNet(nn.Module):
    def __init__(self, **_):
        super(HistogramRestorationNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        # Adjusted residual connections with proper downsampling and channel matching
        self.res1 = nn.Sequential(
            nn.Conv3d(16, 32, 1, stride=1, padding=0),  # Match channels
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),  # Downsample to match size
        )
        self.res2 = nn.Sequential(
            nn.Conv3d(32, 64, 1, stride=1, padding=0),  # Match channels
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),  # Downsample to match size
        )

        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 32 * 32 * 32)
        self.apply(HistogramRestorationNet._init_weights_he)

    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input dimensions: (batch_size, channels(1), 32, 32, 32)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)

        # Apply first adjusted residual connection
        res = self.res1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        x += res  # Add adjusted residual

        # Apply second adjusted residual connection
        res = self.res2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, 2)
        x += res  # Add adjusted residual

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.fc2(x)

        # Reshape back to the histogram shape
        x = x.view(-1, 32, 32, 32)
        x /= torch.sum(x, (1, 2, 3)).view(x.size()[0], 1, 1, 1)

        return x
