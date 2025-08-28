import torch
import torch.nn as nn
import math
from torch.nn import functional as F


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

def BasicNormConv_test():
    # 示例参数
    batch_size = 10  # 批次大小
    channels = 1  # 通道数
    img_H = 128  # 序列长度
    img_W = 128  # 第三维度长度
    kernel_size = 3  # 每个patch的大小 会导致最终 输出的维度为(128/4,128/4)
    dilation = 3  # 膨胀率
    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据
    patch_embedder = BasicNormConv(C_in = channels, C_out = 4, kernel_size = kernel_size, dilation = dilation)
    output = patch_embedder(x)
    print("Output shape:", output.shape)



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


def BasicGhostConv_test():
    # 测试修正后的代码
    ghost_module = BasicGhostConv(inp=2, oup=32, dw_kernel=3, dw_dilated=1)
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
            module = BasicGhostConv(inp, oup, kernel, dilation)
            test_input = torch.randn(1, inp, 32, 32)
            test_output = module(test_input)
            print(f"测试通过: inp={inp}, oup={oup}, kernel={kernel}, dilation={dilation}")
            print(f"  输入形状: {test_input.shape}, 输出形状: {test_output.shape}")
        except Exception as e:
            print(f"测试失败: inp={inp}, oup={oup}, kernel={kernel}, dilation={dilation}")
            print(f"  错误信息: {e}")

def GhostConv2D():
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

def dw_decompos_conv_test():
    # 示例参数
    batch_size = 10  # 批次大小
    channels = 2  # 通道数
    img_H = 256  # 序列长度
    img_W = 256  # 第三维度长度
    dw_kernel = 3  # 每个patch的大小 会导致最终 输出的维度为(128/4,128/4)
    oup = 16  # 每个patch的嵌入维度 嵌入维度即为 卷积输出通道维度
    dw_dilated = 2
    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据
    conv = dw_decompos_conv(inp=channels, oup=oup, dw_kernel=dw_kernel, dw_dilated=dw_dilated)
    output = conv(x)
    print("Input shape:",x.shape)
    print("Output shape:", output.shape)



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

def PatchEmbedding_test():
    # 示例参数
    batch_size = 10  # 批次大小
    channels = 1  # 通道数
    img_H = 128  # 序列长度
    img_W = 128  # 第三维度长度
    patch_size = 4  # 每个patch的大小 会导致最终 输出的维度为(128/4,128/4)
    embed_dim = 16  # 每个patch的嵌入维度 嵌入维度即为 卷积输出通道维度
    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据
    patch_embedder = PatchEmbedding2D(in_channels=channels, patch_size=patch_size, C_out=embed_dim)
    output = patch_embedder(x)
    print("Input shape:",x.shape)
    print("Output shape:", output.shape)




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

def PyramidConvCompress_test():
    # 示例参数
    batch_size = 4  # 批次大小
    channels = 2  # 通道数
    img_H = 256  # 序列长度
    img_W = 256  # 第三维度长度
    img_size = 256

    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据
    compress = PyramidConvCompress(img_size = img_size, C_in = channels, down_sample_size = 8)
    output = compress(x)
    print("Input shape:",x.shape)
    print("Output shape:", output.shape)


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

def Conv_Down_test():
    input_tensor = torch.randn(16, 16, 128, 128)
    DownSamp_exm = Conv_DownSampling2D(16)
    output_tensor = DownSamp_exm(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

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

def Dila_Down_test():
    input_tensor = torch.randn(16, 16, 128, 128)
    DownSamp_exm = Dila_DownSampling2D(16)
    output_tensor = DownSamp_exm(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

class Avg_DownSampling2D(nn.Module):
    def __init__(self, C):
        super(Avg_DownSampling2D, self).__init__()
        self.Down = nn.Sequential(
            # 使用平均池化进行下采样，stride=2表示宽高各减半
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

def Avg_Down_test():
    input_tensor = torch.randn(16, 16, 128, 128)
    DownSamp_exm = Avg_DownSampling2D(16)
    output_tensor = DownSamp_exm(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

class bilinear_UpSam(nn.Module):

    def __init__(self, C):
        super(bilinear_UpSam, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)
    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.Up(up)
        return x

def bilinear_UpSam_test():
    input_tensor = torch.randn(16, 16, 16, 16)
    UpSam = bilinear_UpSam(16)
    output_tensor = UpSam(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")


# 定义一个简单的PixelShuffle层   通道除以四
class PixelShuffle_UpSam(nn.Module):
    # 通道数会除以4
    def __init__(self, upscale_factor=2):
        super(PixelShuffle_UpSam, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(x)

def PixelShuffle_test():

    batch_size = 12
    channels = 4
    height = 8
    width = 8
    upscale_factor = 2

    x = torch.randn(batch_size, channels, height, width)
    print("Input shape:", x.shape)
    pixel_shuffle = PixelShuffle_UpSam(upscale_factor=upscale_factor)
    output = pixel_shuffle(x)
    print("Output shape:", output.shape)

# 定义一个简单的ConvTranspose2d层
class ConvTranspose_UpSam(nn.Module):
    def __init__(self, C):
        super(ConvTranspose_UpSam, self).__init__()
        # 使用反卷积进行上采样，kernel_size=4, stride=2, padding=1
        self.conv_transpose = nn.ConvTranspose2d(C, C // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv_transpose(x)

def ConvTranspose_test():

    batch_size = 4
    channels = 16
    height = 8
    width = 8
    x = torch.randn(batch_size, channels, height, width)
    print("Input shape:", x.shape)
    conv_transpose = ConvTranspose_UpSam(C=channels)
    output = conv_transpose(x)
    print("Output shape:", output.shape)


    


if __name__ == '__main__':
    import time

    # 记录开始时间
    start_time = time.time()

    BasicGhostConv_test()
    # PatchEmbedding_test()
       
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    execution_time = end_time - start_time
    print(f"程序运行时间: {execution_time:.5f} 秒")