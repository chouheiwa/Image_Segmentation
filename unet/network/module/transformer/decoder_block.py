import torch
from torch import nn

from unet.network.module.conv_base_block import ConvBaseBlock


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = ConvBaseBlock(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.conv2 = ConvBaseBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
