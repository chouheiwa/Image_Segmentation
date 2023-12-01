import torch
from torch import nn

from unet.utils import np2th
from .std_conv_2d import StdConv2d1x1, StdConv2d3x3
from os.path import join


class PreActBottleneck(nn.Module):
    """
    Pre-activation (v2) bottleneck block.
    """

    def __init__(self, channel_input: int, channel_output: int = None, channel_middle: int = None, stride: int = 1):
        super().__init__()
        channel_output = channel_output or channel_input
        channel_middle = channel_middle or channel_output // 4

        self.gn1 = nn.GroupNorm(32, channel_middle, eps=1e-6)
        self.conv1 = StdConv2d1x1(channel_input, channel_middle, bias=False)
        self.gn2 = nn.GroupNorm(32, channel_middle, eps=1e-6)
        self.conv2 = StdConv2d3x3(channel_middle, channel_middle, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, channel_output, eps=1e-6)
        self.conv3 = StdConv2d1x1(channel_middle, channel_output, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or channel_input != channel_output:
            # Projection also with pre-activation according to paper.
            self.downsample = StdConv2d1x1(channel_input, channel_output, stride, bias=False)
            self.gn_proj = nn.GroupNorm(channel_output, channel_output)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[join(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[join(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[join(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[join(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[join(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[join(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[join(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[join(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[join(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[join(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[join(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[join(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))
