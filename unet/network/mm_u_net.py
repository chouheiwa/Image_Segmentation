import torch
from torch import nn

from unet.network.module import ConvBlock, UpConv, MonarchMixerLayer


class MMUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, sqrt_n_values=None, sqrt_d_values=None):
        super(MMUNet, self).__init__()

        # 定义 U-Net 的组件
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.Up5 = UpConv(ch_in=1024, ch_out=512)
        self.Up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.Up4 = UpConv(ch_in=512, ch_out=256)
        self.Up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.Up3 = UpConv(ch_in=256, ch_out=128)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.Up2 = UpConv(ch_in=128, ch_out=64)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        # 定义 MonarchMixerLayer 实例
        if sqrt_n_values is None:
            sqrt_n_values = [16, 16, 16, 16]  # 示例值，需要根据实际情况调整
        if sqrt_d_values is None:
            sqrt_d_values = [16, 16, 16, 16]  # 示例值，需要根据实际情况调整

        self.mixer4 = MonarchMixerLayer(sqrt_n_values[0], sqrt_d_values[0])
        self.mixer3 = MonarchMixerLayer(sqrt_n_values[1], sqrt_d_values[1])
        self.mixer2 = MonarchMixerLayer(sqrt_n_values[2], sqrt_d_values[2])

    def forward(self, x):
        # 编码器路径
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # 解码器 + 连接路径
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.mixer4(d4)  # 使用 MonarchMixerLayer

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.mixer3(d3)  # 使用 MonarchMixerLayer

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.mixer2(d2)  # 使用 MonarchMixerLayer

        d1 = self.Conv_1x1(d2)

        return d1
