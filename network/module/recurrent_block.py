import torch.nn as nn


class RecurrentBlock(nn.Module):
    """
    循环卷积块。这个模块通过重复应用相同的卷积操作来加强特征图的特征。
    它执行给定次数的卷积操作，每次都将前一次的输出与原始输入相加，
    以此增强特征并保持原始信息。

    参数:
    - ch_out (int): 输入和输出特征图的通道数。由于是循环卷积，输入和输出的通道数相同。
    - t (int): 循环卷积的次数，默认为2。
    """

    def __init__(self, ch_out, t=2):
        """
        初始化循环卷积块。

        参数:
        - ch_out (int): 输入和输出特征图的通道数。
        - t (int): 循环卷积的次数，默认为2。
        """
        super(RecurrentBlock, self).__init__()
        self.t = t  # 循环次数
        self.ch_out = ch_out  # 通道数

        # 定义卷积操作
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播方法，定义了数据如何通过网络。

        参数:
        - x: 输入特征图

        返回:
        - x1: 经过循环卷积处理后的特征图
        """
        x1 = self.conv(x)
        for i in range(self.t - 1):
            x1 = self.conv(x + x1)  # 后续循环中，将前一次的输出与原始输入相加后再卷积

        return x1  # 返回最终的特征图
