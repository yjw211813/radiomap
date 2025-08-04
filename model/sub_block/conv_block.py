import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class GhostConv2D(nn.Module):
    def __init__(self, inp, oup,depth_wise_size,dilated_num , channel_kernel_size=1, ratio=2, stride=1, relu=True):
        super(GhostConv2D, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels = inp, out_channels = init_channels, kernel_size = channel_kernel_size,stride = stride,padding = channel_kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 计算扩张后的卷积核大小
        dilated_kernel_size = (depth_wise_size - 1) * dilated_num + 1
        # 计算padding，确保输出宽度与输入宽度相同
        padding_width = (dilated_kernel_size - 1) // 2
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=depth_wise_size, stride=(1, 1),
                      dilation=dilated_num, padding=padding_width, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

def GhostConv2D_test():
    input_tensor = torch.randn(10, 2, 128, 128)
    # 创建 GhostModule 实例
    ghost_module = GhostConv2D(inp=2, oup=32,depth_wise_size=3,dilated_num=1 )
    # 进行前向传播
    output_tensor = ghost_module(input_tensor)
    # 打印输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")


class PatchEmbedding2D(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding2D, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 卷积操作，使用1x8的卷积核对第三维度进行patch划分
        # input size: (batch_size, 1, 128, 128)
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride= patch_size)

    def forward(self, x):
        # x 的形状是 (batch_size, channels, 128, 128)
        x = self.conv(x)  # 应用卷积，划分patch
        # 形状变化为 (batch_size, embed_dim, 128, num_patches)
        return x

def PatchEmbedding_test():
    # 示例参数
    batch_size = 10  # 批次大小
    channels = 1  # 通道数
    img_H = 128  # 序列长度
    img_W = 128  # 第三维度长度
    patch_size = 4  # 每个patch的大小 会导致最终 输出的维度为(128/4,128/4)
    embed_dim = 16  # 每个patch的嵌入维度 嵌入维度即为 卷积输出通道维度
    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据
    patch_embedder = PatchEmbedding2D(in_channels=channels, patch_size=patch_size, embed_dim=embed_dim)
    output = patch_embedder(x)
    print("Output shape:", output.shape)



if __name__ == '__main__':
    # GhostConv2D_test()
    PatchEmbedding_test()
