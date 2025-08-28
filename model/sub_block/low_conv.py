import torch
import torch.nn as nn
import math
from torch.nn import functional as F


# deepseek 推荐先 BatchNorm2d 再进行 GELU

class BasicNormConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, dilation = 1, dropout_rate=0.05,groups = 1,gelu=True):
        super(BasicNormConv, self).__init__()
        dilated_kernel_size = (kernel_size - 1) * dilation + 1
        padding_width = (dilated_kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size,
                      dilation=dilation, padding=padding_width,groups=groups),
            nn.BatchNorm2d(C_out),
            nn.GELU() if gelu else nn.Sequential(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.block(x)

class BasicGhostConv(nn.Module):
    def __init__(self, inp, oup, dw_kernel, dw_dilated, channel_kernel_size=1, ratio=2):
        super(BasicGhostConv, self).__init__()
        self.oup = oup

        # ratiao 是 第二次分组卷积相对于1*1通道卷积的比例

        init_channels = math.ceil(oup / ratio) # 1*1通道卷积 输出通道数
        # 确保 new_channels 是 init_channels 的整数倍
        new_channels = self.cal_dwconv_out(init_channels, ratio)

        # 先进行1*1卷积
        self.primary_conv = BasicNormConv(C_in = inp, 
                                          C_out = init_channels, 
                                          kernel_size = channel_kernel_size,)
        # 再进行卷积核卷积
        self.cheap_operation = BasicNormConv(C_in = init_channels, 
                                          C_out = new_channels, 
                                          kernel_size = dw_kernel, 
                                          dilation = dw_dilated,
                                          groups = init_channels)


    def cal_dwconv_out(self,init_channels, ratio):
        """
        计算输出卷积通道数。
        参数：
        - init_channels: 初始化的卷积通道数
        - ratio: 通道扩展的比例
        返回：
        - output_channels: 计算后的输出通道数
        """
        # 计算新的浮动通道数
        new_channels_float = init_channels * (ratio - 1)
        multiple = round(new_channels_float / init_channels)
        new_channels = init_channels * multiple

        # 确保总通道数至少为 oup
        total_channels = init_channels + new_channels
        if total_channels < self.oup:
            # 增加倍数以满足要求
            multiple += math.ceil((self.oup - total_channels) / init_channels)
            # 确保 new_channels 是 init_channels 的整数倍
            new_channels = init_channels * multiple

        # 返回最终计算的输出通道数
        return new_channels

    def forward(self, x):
        x1 = self.primary_conv(x)

        if self.cheap_operation is not None:
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x1

        return out[:, :self.oup, :, :]

class GhostConv2D(nn.Module):

    def __init__(self, inp, oup, dw_kernel, dw_dilated, channel_kernel_size=1, ratio=2, drop_out=0.05):
        super(GhostConv2D, self).__init__()
        self.ghost_conv = BasicGhostConv(inp=inp,
                                         oup=oup,
                                         dw_kernel=dw_kernel,
                                         dw_dilated=dw_dilated,
                                         channel_kernel_size=channel_kernel_size,
                                         ratio=ratio)
        self.norm = nn.BatchNorm2d(oup)
        self.drop = nn.Dropout(drop_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.ghost_conv(x)  # 通过Ghost卷积层
        x = self.norm(x)        # 应用批量归一化
        x = self.act(x)         # 应用激活函数
        x = self.drop(x)        # 应用dropout
        return x


class dw_decompos_conv(nn.Module):
    def __init__(self, inp, oup, dw_kernel, dw_dilated):
        super(dw_decompos_conv, self).__init__()

        self.conv1X1 = BasicNormConv(C_in = inp, C_out = oup, kernel_size = 1, dilation = 1)
        self.dw_conv = BasicNormConv(C_in = oup, C_out = oup, kernel_size = dw_kernel, dilation = dw_dilated, groups = oup)
    def forward(self, x):
        out = self.dw_conv(self.conv1X1(x))
        return out


class PatchEmbedding2D(nn.Module):
    def __init__(self, in_channels, patch_size, C_out):
        super(PatchEmbedding2D, self).__init__()
        # 卷积操作，使用1x8的卷积核对第三维度进行patch划分
        # input size: (batch_size, 1, 128, 128)
        self.conv = nn.Conv2d(in_channels, C_out, kernel_size=patch_size, stride= patch_size)

    def forward(self, x):
        # x 的形状是 (batch_size, channels, 128, 128)
        x = self.conv(x)  # 应用卷积，划分patch
        # 形状变化为 (batch_size, embed_dim, 128, num_patches)
        return x

class PyramidConvCompress(nn.Module):
    def __init__(self,img_size, C_in, down_sample_size):
        super(PyramidConvCompress, self).__init__()
        
        self.compress_info = nn.ModuleList()  # 用于存储不同层的卷积模块

        # 当前特征图大小
        current_size = img_size 
        
        # 添加分层卷积降维直到达到目标尺寸
        while current_size > down_sample_size:
            self.compress_info.append(
                nn.Sequential(
                    nn.Conv2d(C_in, C_in, kernel_size=3, stride=2, groups=C_in, padding=1),
                    nn.GELU()
                )
            )
            current_size = current_size // 2  # 每次降采样尺寸减半

        # 最后一个卷积层，通常会使用较大的卷积核来整合特征
        self.compress_info.append(nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=current_size, groups=C_in),
            nn.GELU()
        ))

    def forward(self, x):
        # 前向传播
        for layer in self.compress_info:
            x = layer(x)  # 依次通过每个卷积模块
        return x

class Conv_DownSampling2D(nn.Module):
    def __init__(self, C):
        super(Conv_DownSampling2D, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.Down(x)


class Dila_DownSampling2D(nn.Module):
    def __init__(self, C):
        super(Dila_DownSampling2D, self).__init__()
        dila_num = 2
        self.Down = nn.Sequential(
            # 使用扩张卷积进行下采样
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=dila_num, dilation=dila_num),
            nn.BatchNorm2d(C),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

# 定义一个简单的PixelShuffle层   通道除以四
class PixelShuffle_UpSam(nn.Module):
    # 通道数会除以4
    def __init__(self, upscale_factor=2):
        super(PixelShuffle_UpSam, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(x)

# 定义一个简单的ConvTranspose2d层
class ConvTranspose_UpSam(nn.Module):
    def __init__(self, C):
        super(ConvTranspose_UpSam, self).__init__()
        # 使用反卷积进行上采样，kernel_size=4, stride=2, padding=1
        self.conv_transpose = nn.ConvTranspose2d(C, C // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv_transpose(x)

