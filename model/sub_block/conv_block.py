import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class GhostConv2D(nn.Module):
    def __init__(self, inp, oup, depth_wise_size, dilated_num, channel_kernel_size=1, ratio=3.5, stride=1, relu=True):
        super(GhostConv2D, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)

        # 确保 new_channels 是 init_channels 的整数倍
        # 计算最接近的整数倍
        new_channels_float = init_channels * (ratio - 1)
        multiple = round(new_channels_float / init_channels)
        new_channels = init_channels * multiple

        # 确保总通道数至少为 oup
        total_channels = init_channels + new_channels
        if total_channels < oup:
            # 增加倍数以满足要求
            multiple += math.ceil((oup - total_channels) / init_channels)
            # 确保 new_channels 是 init_channels 的整数倍
            new_channels = init_channels * multiple


        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=init_channels,
                      kernel_size=channel_kernel_size, stride=stride,
                      padding=channel_kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 计算扩张后的卷积核大小
        dilated_kernel_size = (depth_wise_size - 1) * dilated_num + 1
        # 计算padding，确保输出宽度与输入宽度相同
        padding_width = (dilated_kernel_size - 1) // 2

        # 只有在 new_channels > 0 时才创建廉价操作
        if new_channels > 0:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, kernel_size=depth_wise_size,
                          stride=(1, 1), dilation=dilated_num, padding=padding_width,
                          groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        else:
            self.cheap_operation = None

    def forward(self, x):
        x1 = self.primary_conv(x)

        if self.cheap_operation is not None:
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x1

        return out[:, :self.oup, :, :]


def GhostConv2D_test():
    # 测试修正后的代码
    ghost_module = GhostConv2D(inp=2, oup=32, depth_wise_size=3, dilated_num=1)
    print("GhostConv2D 创建成功")

    # 测试前向传播
    x = torch.randn(1, 2, 64, 64)
    output = ghost_module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 测试不同参数组合
    test_cases = [
        (3, 64, 5, 2),
        (1, 16, 3, 1),
        (4, 128, 7, 3),
    ]

    for inp, oup, kernel, dilation in test_cases:
        try:
            module = GhostConv2D(inp, oup, kernel, dilation)
            test_input = torch.randn(1, inp, 32, 32)
            test_output = module(test_input)
            print(f"测试通过: inp={inp}, oup={oup}, kernel={kernel}, dilation={dilation}")
            print(f"  输入形状: {test_input.shape}, 输出形状: {test_output.shape}")
        except Exception as e:
            print(f"测试失败: inp={inp}, oup={oup}, kernel={kernel}, dilation={dilation}")
            print(f"  错误信息: {e}")



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
    # PatchEmbedding_test()
    GhostConv2D_test()