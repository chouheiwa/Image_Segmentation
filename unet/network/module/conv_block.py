from torch import nn
from .conv_base_block import ConvBaseBlock


class ConvBlock(nn.Sequential):
    def __init__(
            self,
            ch_in,
            ch_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
    ):
        cov_1 = ConvBaseBlock(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        conv_2 = ConvBaseBlock(
            ch_out,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        super(ConvBlock, self).__init__(cov_1, conv_2)
