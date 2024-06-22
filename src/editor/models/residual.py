import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, **_):
        super(Residual, self).__init__()

        # Assuming the input histograms are 3D tensors of shape (bin_count, bin_count, bin_count)
        # Convolutional layers to extract features from the histograms
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)

        # Batch normalization layers for better convergence
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)

        # Residual block to help the network learn identity functions effectively
        self.resblock1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
        )

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Apply residual blocks
        residual = x
        out = self.resblock1(x)
        out += residual
        out = self.relu(out)

        # Upsample to original size
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))

        return out
