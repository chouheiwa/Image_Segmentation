import torch
from torch import nn
from torch.nn import functional


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return functional.conv2d(
            x, w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class StdConv2d1x1(StdConv2d):
    def __init__(self, c_in, c_out, stride=1, bias=False):
        super().__init__(
            c_in,
            c_out,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=bias
        )


class StdConv2d3x3(StdConv2d):
    def __init__(self, c_in, c_out, stride=1, bias=False):
        super().__init__(
            c_in,
            c_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias
        )
