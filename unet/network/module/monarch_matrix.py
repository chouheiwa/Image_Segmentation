from einops import rearrange
import torch
from torch import nn


def blockdiag_matmul(x, w):
    return torch.einsum(
        "bnm ,...bm ->...bn", w, x.view(*x.shape[:-1], w.shape[0], w.shape[-1])
    ).reshape(*x.shape)


class MonarchMatrix(nn.Module):
    def __init__(self, sqrt_n: int):
        super().__init__()
        self.sqrt_n = sqrt_n
        self.L = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))
        self.R = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))

    def forward(self, x):
        # 对于输入张量 x，它保持除了最后一个维度之外的所有维度不变，并且重新排列最后一个维度，使得原本相邻的 m 和 n 子维度变成交错排列。
        # 这种操作在处理多维数据时非常有用，特别是在深度学习和数组处理中，用于改变数据的形状以满足特定的网络架构或计算需求。
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.L)
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.R)
        return rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)


class MonarchMixerLayer(nn.Module):
    def __init__(self, sqrt_n: int, sqrt_d: int):
        super().__init__()
        self.m1 = MonarchMatrix(sqrt_n)
        self.m2 = MonarchMatrix(sqrt_n)
        self.m3 = MonarchMatrix(sqrt_d)
        self.m4 = MonarchMatrix(sqrt_d)

        self.n_kernel = nn.Parameter(torch.randn(sqrt_d ** 2, sqrt_n ** 2))
        self.d_kernel = nn.Parameter(torch.randn(1, sqrt_d ** 2))
        self.layer_norm = nn.LayerNorm(sqrt_d ** 2)

    def forward(self, x: torch.Tensor):  # x.shape = (b, n, d)
        print(f"Input Shape: {x.shape}")
        x_reshape = x.view(1, -1, 256)
        x_tilde = self.m2(torch.relu(self.n_kernel * self.m1(x.transpose(-1, -2)))).transpose(-1, -2)  # mix sequence
        y = self.m4(torch.relu(self.d_kernel * self.m3(x_tilde)))  # mix features
        return self.layer_norm(y + x_tilde)  # skip connection
    # def forward(self, x: torch.Tensor):
    #     # 假设 x 的形状是 (batch_size, channels, height, width)
    #     batch_size, channels, height, width = x.shape
    #
    #     # 将 x 变形以适配 MonarchMixerLayer
    #     # 这里我们将 height 和 width 维度合并，并保持 channels 维度不变
    #     x = x.view(batch_size, channels, -1)
    #
    #     # 执行 MonarchMixerLayer 的操作
    #     x_tilde = self.m2(torch.relu(self.n_kernel * self.m1(x.transpose(-1, -2)))).transpose(-1, -2)
    #     y = self.m4(torch.relu(self.d_kernel * self.m3(x_tilde)))
    #
    #     # 变形回原始的四维形状
    #     y = y.view(batch_size, channels, height, width)
    #
    #     # 应用层归一化和跳过连接
    #     return self.layer_norm(y + x)  # 注意：这里可能需要调整


if __name__ == '__main__':
    n = 3328
    d = 64
    x = torch.randn(1, n, d)  # 假设这是你的原始张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mixer = MonarchMixerLayer(n, d)  # 创建 MonarchMixerLayer 实例
    mixer.to(device)
    y = mixer(x)  # 执行 MonarchMixerLayer 的前向传播
    print(f"Out shape: {y.shape}")
