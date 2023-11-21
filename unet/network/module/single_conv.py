import torch.nn as nn


class SingleConv(nn.Module):
    """
    单层卷积块。这个模块包含一个二维卷积层，后面跟着一个批量归一化层和ReLU激活函数。
    这是一个在神经网络中常见的基础结构，用于提取特征。

    参数:
    - ch_in (int): 输入特征图的通道数。
    - ch_out (int): 输出特征图的通道数。
    """
    def __init__(self, ch_in, ch_out):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
