from collections import OrderedDict

import torch
from torch import nn
from .std_conv_2d import StdConv2d
from .pre_act_bottleneck import PreActBottleneck


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(channel_input=width, channel_output=width * 4, channel_middle=width))] +
                [(f'unit{i:d}',
                  PreActBottleneck(channel_input=width * 4, channel_output=width * 4, channel_middle=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(channel_input=width * 4, channel_output=width * 8, channel_middle=width * 2,
                                            stride=2))] +
                [(f'unit{i:d}',
                  PreActBottleneck(channel_input=width * 8, channel_output=width * 8, channel_middle=width * 2)) for i
                 in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1',
                  PreActBottleneck(channel_input=width * 8, channel_output=width * 16, channel_middle=width * 4,
                                   stride=2))] +
                [(f'unit{i:d}',
                  PreActBottleneck(channel_input=width * 16, channel_output=width * 16, channel_middle=width * 4)) for i
                 in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert 3 > pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
