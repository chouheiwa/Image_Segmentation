import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    注意力机制块。这个模块通过计算输入特征的注意力权重，来强调重要的特征并抑制不重要的特征。
    它包含了两个卷积层来处理两种不同的特征图，并通过一个激活函数和sigmoid函数计算注意力权重。

    参数:
    - F_g (int): 全局特征图的通道数。
    - F_l (int): 局部特征图的通道数。
    - F_int (int): 中间特征图的通道数，用于调整通道数以便于后续操作。
    """

    def __init__(self, F_g, F_l, F_int):
        """
        初始化注意力机制块。

        参数:
        - F_g (int): 全局特征图的通道数。
        - F_l (int): 局部特征图的通道数。
        - F_int (int): 中间特征图的通道数。
        """
        super(AttentionBlock, self).__init__()
        # 用于处理全局特征图的卷积层
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 用于处理局部特征图的卷积层
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 计算注意力权重的卷积层
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        前向传播方法，定义了数据如何通过网络。

        参数:
        - g: 全局特征图。
        - x: 局部特征图。

        返回:
        - 经过注意力加权的特征图。
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 将处理过的全局特征图和局部特征图相加并应用ReLU激活函数
        psi = self.relu(g1 + x1)
        # 计算注意力权重
        psi = self.psi(psi)
        # 将注意力权重应用于局部特征图
        return x * psi
