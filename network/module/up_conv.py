from torch import nn


class UpConv(nn.Module):
    """
    上采样和卷积模块。这个模块首先将输入的特征图大小上采样两倍，然后应用一个二维卷积。
    此外，还包括了批量归一化和ReLU激活函数。

    参数:
    - ch_in (int): 输入特征图的通道数。
    - ch_out (int): 输出特征图的通道数。
    """
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)
