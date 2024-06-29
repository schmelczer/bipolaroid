import logging
import torch
import torch.nn as nn


EPSILON = 1e-5


class HistogramNet(nn.Module):
    def __init__(
        self,
        elu_alpha: float = 1,
        dropout_prob: float = 0.05,
        features: list[int] = [16, 32, 64],
        kernel_size: int = 3,
        use_instance_norm: bool = True,
        use_elu: bool = True,
        leaky_relu_alpha: float = 0.01,
        use_residual: bool = True,
        **_,
    ):
        super(HistogramNet, self).__init__()
        self._elu_alpha = elu_alpha
        self._dropout_prob = dropout_prob
        self._features = features
        self._kernel_size = kernel_size
        self._use_instance_norm = use_instance_norm
        self._use_elu = use_elu
        self._leaky_relu_alpha = leaky_relu_alpha
        self._use_residual = use_residual

        self._convolutions = nn.ModuleList(
            self._make_conv_layer(in_channels=in_channels, out_channels=out_channels)
            for in_channels, out_channels in zip([1] + features[:-1], features)
        )

        if self._use_residual:
            self._residual_blocks = nn.ModuleList(
                self._make_resblock(channels) for channels in features
            )

        self._deconvolutions = nn.ModuleList(
            self._make_deconv_layer(in_channels=in_channels, out_channels=out_channels)
            for in_channels, out_channels in zip(
                features[::-1], features[::-1][1:] + [1]
            )
        )

        self._initialize_weights()

    def _make_conv_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self._kernel_size,
                padding=self._kernel_size // 2,
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

    def _make_resblock(self, channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=self._kernel_size,
                padding=self._kernel_size // 2,
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
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=self._kernel_size,
                padding=self._kernel_size // 2,
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

    def _make_deconv_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self._kernel_size,
                padding=self._kernel_size // 2,
            ),
            (
                nn.ELU(self._elu_alpha)
                if self._use_elu
                else nn.LeakyReLU(self._leaky_relu_alpha)
            ),
        )

    def forward(self, x):
        if self._use_residual:
            for conv, res in zip(self._convolutions, self._residual_blocks):
                x = conv(x)
                x = x + res(x)
        else:
            for conv in self._convolutions:
                x = conv(x)

        for deconv in self._deconvolutions:
            x = deconv(x)

        return self._normalize(x)

    @staticmethod
    def _normalize(x):
        x = torch.clamp(x, min=0)
        x_sum = torch.sum(x, dim=(2, 3, 4), keepdim=True)
        return x / (x_sum + EPSILON)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
