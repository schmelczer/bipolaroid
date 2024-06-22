import torch
import torch.nn as nn


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Residual3(nn.Module):
    def __init__(
        self,
        elu_alpha: float = 1,
        dropout_prob: float = 0.1,
        use_depthwise_separable_conv: bool = False,
        feature_map_sizes: list[int] = [16, 32, 64],
        kernel_sizes: list[int] = [3, 3, 3],
    ):
        super(Residual3, self).__init__()

        conv = DepthwiseSeparableConv3d if use_depthwise_separable_conv else nn.Conv3d

        # Assuming the input histograms are 3D tensors of shape (bin_count, bin_count, bin_count)
        # Convolutional layers to extract features from the histograms
        self.conv1 = conv(
            1, feature_map_sizes[0], kernel_size=kernel_sizes[0], padding=1
        )
        self.conv2 = conv(
            feature_map_sizes[0],
            feature_map_sizes[1],
            kernel_size=kernel_sizes[1],
            padding=1,
        )
        self.conv3 = conv(
            feature_map_sizes[1],
            feature_map_sizes[2],
            kernel_size=kernel_sizes[2],
            padding=1,
        )

        self.activation = nn.ELU(elu_alpha, inplace=True)

        self.bn1 = nn.BatchNorm3d(feature_map_sizes[0])
        self.bn2 = nn.BatchNorm3d(feature_map_sizes[1])
        self.bn3 = nn.BatchNorm3d(feature_map_sizes[2])

        self.resblock1 = nn.Sequential(
            conv(
                feature_map_sizes[2],
                feature_map_sizes[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ELU(elu_alpha, inplace=True),
            nn.BatchNorm3d(feature_map_sizes[2]),
            conv(
                feature_map_sizes[2],
                feature_map_sizes[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ELU(elu_alpha, inplace=True),
            nn.BatchNorm3d(feature_map_sizes[2]),
        )

        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose3d(
            feature_map_sizes[2],
            feature_map_sizes[1],
            kernel_size=feature_map_sizes[2],
            stride=1,
            padding=1,
        )
        self.deconv2 = nn.ConvTranspose3d(
            feature_map_sizes[1],
            feature_map_sizes[0],
            kernel_size=feature_map_sizes[1],
            stride=1,
            padding=1,
        )
        self.deconv3 = nn.ConvTranspose3d(
            feature_map_sizes[0], 1, kernel_size=3, stride=1, padding=1
        )

        self.dropout = nn.Dropout3d(p=dropout_prob)
        self._initialize_weights()

    def forward(self, x):
        out = self.dropout(self.bn1(self.activation(self.conv1(x))))
        out = self.dropout(self.bn2(self.activation(self.conv2(out))))
        out = self.dropout(self.bn3(self.activation(self.conv2(out))))

        out = out + self.resblock1(out)

        out = self.activation(self.deconv1(out))
        out = self.activation(self.deconv2(out))
        out = self.activation(self.deconv3(out))

        return self._normalize(out)

    def _normalize(self, x):
        x_sum = torch.sum(x, dim=(2, 3, 4), keepdim=True)
        return x / torch.where(x_sum == 0, torch.ones_like(x_sum), x_sum)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(
                    m.weight,
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def _test_network_dimensions(constructor):
    for bin_count in [16, 32, 64]:
        model = constructor()

        # Create a dummy input tensor of the correct shape
        input_tensor = torch.rand(4, 1, bin_count, bin_count, bin_count)

        # Test the model output
        output = model(input_tensor)
        assert (
            input_tensor.shape == output.shape
        ), f"Expected output shape {input_tensor.shape}, but got {output.shape}"


_test_network_dimensions(Residual3)
