import torch.nn as nn
import torch.nn.functional as F


class NormalisedCNN(nn.Module):
    def __init__(self, bin_count):
        super(NormalisedCNN, self).__init__()
        self.bin_count = bin_count

        # Define the layers of the neural network
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(32)
        self.conv5 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(16)
        self.conv6 = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.view(
            -1, 1, self.bin_count, self.bin_count, self.bin_count
        )  # Reshape input to (N, C, D, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        return x
