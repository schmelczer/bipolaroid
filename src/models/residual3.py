import logging
import torch
import torch.nn as nn


EPSILON = 1e-5


class Residual3(nn.Module):
    def __init__(
        self,
        elu_alpha: float = 1,
        dropout_prob: float = 0.05,
        features: list[int] = [16, 32, 64],
        kernel_sizes: list[int] = [3, 3, 3],
        use_instance_norm: bool = True,
        use_elu: bool = True,
        leaky_relu_alpha: float = 0.01,
        **_,
    ):
        super(Residual3, self).__init__()
        self._elu_alpha = elu_alpha
        self._dropout_prob = dropout_prob
        self._features = features
        self._kernel_sizes = kernel_sizes
        self._use_instance_norm = use_instance_norm
        self._use_elu = use_elu
        self._leaky_relu_alpha = leaky_relu_alpha
        self.print_og_result = False

        self.conv1 = self._make_conv_layer(1, features[0], kernel_sizes[0])
        self.res1 = self._make_resblock(features[0], kernel_sizes[0])
        self.conv2 = self._make_conv_layer(features[0], features[1], kernel_sizes[1])
        self.res2 = self._make_resblock(features[1], kernel_sizes[1])
        self.conv3 = self._make_conv_layer(features[1], features[2], kernel_sizes[2])
        self.res3 = self._make_resblock(features[2], kernel_sizes[2])

        self.deconv1 = self._make_deconv_layer(
            features[2], features[1], kernel_sizes[2]
        )
        self.deconv2 = self._make_deconv_layer(
            features[1], features[0], kernel_sizes[1]
        )
        self.deconv3 = self._make_deconv_layer(features[0], 1, kernel_sizes[0])

        self._initialize_weights()

    def _make_conv_layer(
        self, in_channels: int, out_channels: int, kernel_size: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            (
                nn.ELU(self._elu_alpha)
                if self._use_elu
                else nn.LeakyReLU(self._leaky_relu_alpha)
            ),
            (nn.InstanceNorm3d if self._use_instance_norm else nn.BatchNorm3d)(
                out_channels
            ),
            nn.Dropout(p=self._dropout_prob),
        )

    def _make_resblock(self, channels: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            (
                nn.ELU(self._elu_alpha)
                if self._use_elu
                else nn.LeakyReLU(self._leaky_relu_alpha)(
                    nn.InstanceNorm3d if self._use_instance_norm else nn.BatchNorm3d
                )(channels)
            ),
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            (
                nn.ELU(self._elu_alpha)
                if self._use_elu
                else nn.LeakyReLU(self._leaky_relu_alpha)
            ),
            (nn.InstanceNorm3d if self._use_instance_norm else nn.BatchNorm3d)(
                channels
            ),
        )

    def _make_deconv_layer(
        self, in_channels: int, out_channels: int, kernel_size: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
            ),
            (
                nn.ELU(self._elu_alpha)
                if self._use_elu
                else nn.LeakyReLU(self._leaky_relu_alpha)
            ),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.res1(out)
        out = self.conv2(out)
        out = out + self.res2(out)
        out = self.conv3(out)
        out = out + self.res3(out)

        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)

        if self.print_og_result:
            logging.info(f"Original result {torch.sum(out)}")
            self.print_og_result = False

        return self._normalize(out)

    @staticmethod
    def _normalize(x):
        x = torch.clamp(x, min=0)
        x_sum = torch.sum(x, dim=(2, 3, 4), keepdim=True)
        return x / (x_sum + EPSILON)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                # Applying He normal initialization
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
