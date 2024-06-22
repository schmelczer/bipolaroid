import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, **_):
        super(SimpleCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv3d(
            1, 16, kernel_size=3, padding=1
        )  # input channels = 1, output channels = 16
        self.conv2 = nn.Conv3d(
            16, 32, kernel_size=3, padding=1
        )  # input channels = 16, output channels = 32
        self.conv3 = nn.Conv3d(
            32, 64, kernel_size=3, padding=1
        )  # input channels = 32, output channels = 64
        self.conv4 = nn.Conv3d(
            64, 32, kernel_size=3, padding=1
        )  # input channels = 64, output channels = 32
        self.conv5 = nn.Conv3d(
            32, 16, kernel_size=3, padding=1
        )  # input channels = 32, output channels = 16
        self.conv6 = nn.Conv3d(
            16, 1, kernel_size=3, padding=1
        )  # input channels = 16, output channels = 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        sum = torch.sum(x, dim=(2, 3, 4), keepdim=True)
        return x / sum
