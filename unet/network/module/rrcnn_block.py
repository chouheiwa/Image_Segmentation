import torch.nn as nn

from unet.network.module import RecurrentBlock


class RRCNNBlock(nn.Module):
    """
    RRCNN（递归卷积神经网络）块。这个模块首先通过一个1x1卷积来调整特征图的通道数，
    然后应用两个循环卷积块来加强特征提取。最后，它将原始输入（经过1x1卷积处理）和
    经过循环卷积处理的输出进行相加，以此结合原始信息和加强特征。

    参数:
    - ch_in (int): 输入特征图的通道数。
    - ch_out (int): 输出特征图的通道数。
    - t (int): 每个循环卷积块内的循环次数，默认为2。
    """

    def __init__(self, ch_in, ch_out, t=2):
        """
        初始化RRCNN块。

        参数:
        - ch_in (int): 输入特征图的通道数。
        - ch_out (int): 输出特征图的通道数。
        - t (int): 每个循环卷积块内的循环次数，默认为2。
        """
        super(RRCNNBlock, self).__init__()
        # 定义两个循环卷积块
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        # 定义1x1卷积，用于调整通道数
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播方法，定义了数据如何通过网络。

        参数:
        - x: 输入特征图

        返回:
        - 输出特征图，结合了原始输入和循环卷积处理的结果
        """
        # 通过1x1卷积调整特征图的通道数
        x = self.Conv_1x1(x)
        # 应用循环卷积块
        x1 = self.RCNN(x)
        # 将原始输入（经过1x1卷积处理）和循环卷积处理的输出相加
        return x + x1
