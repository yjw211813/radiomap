import torch
import torch.nn as nn
from sympy.strategies.core import switch
import math


# 定义一个 Patch Embedding 模块
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 卷积操作，使用1x8的卷积核对第三维度进行patch划分
        # input size: (batch_size, 1, 2, 128)
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x):
        # x 的形状是 (batch_size, channels, 2, 128)
        x = self.conv(x)  # 应用卷积，划分patch
        # 形状变化为 (batch_size, embed_dim, 2, num_patches)
        return x

def test_PatchEmbedding():
    # 示例参数
    batch_size = 10  # 批次大小
    channels = 2  # 通道数
    seq_len = 1  # 序列长度
    feature_len = 128  # 第三维度长度

    patch_size = 4  # 每个patch的大小
    embed_dim = 32  # 每个patch的嵌入维度

    # 创建一个模拟的输入数据
    x = torch.randn(batch_size, channels, seq_len, feature_len)  # 随机生成输入数据

    # 创建 PatchEmbedding 模块
    patch_embedder = PatchEmbedding(in_channels=channels, patch_size=patch_size, embed_dim=embed_dim)

    # 前向传播
    output = patch_embedder(x)

    # 打印输出的形状
    print("Output shape:", output.shape)

class GhostModule(nn.Module):
    def __init__(self, inp, oup,depth_wise_size,dilated_num , channel_kernel_size=1, ratio=2, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, channel_kernel_size, stride, channel_kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 计算扩张后的卷积核大小
        dilated_kernel_size = (depth_wise_size - 1) * dilated_num + 1
        # 计算padding，确保输出宽度与输入宽度相同
        padding_width = (dilated_kernel_size - 1) // 2
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=(1, depth_wise_size), stride=(1, 1),
                      dilation=dilated_num, padding=(0, padding_width), groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

def ghost_test():
    input_tensor = torch.randn(1, 2, 1, 128)
    # 创建 GhostModule 实例
    ghost_module = GhostModule(inp=2, oup=31,depth_wise_size=3,dilated_num=1 )
    # 进行前向传播
    output_tensor = ghost_module(input_tensor)
    # 打印输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.GELU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).reshape(b, c)  # 使用 reshape 替代 view
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).reshape(b, c, 1, 1)  # 使用 reshape 替代 view
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

def se_test():
    # 测试 SE_Block
    x = torch.randn(2, 32, 1, 128)  # 假设输入是一个 batch_size 为 2，通道数为 32，8x8 的特征图
    se_block = SE_Block(inchannel=32)
    output = se_block(x)
    print(output.shape)  # 应输出 (2, 32, 8, 8)

class Inception_group(nn.Module):
    def __init__(self, C_in, C_out,kernel_sizes,dilated_num):
        super(Inception_group, self).__init__()

        if C_out % 4 != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by 4.")

        sub_Cout = int(C_out / 4)

        kernel_size = kernel_sizes[0]
        dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
        padding_width = (dilated_kernel_size - 1) // 2
        gcd_value = math.gcd(C_in, sub_Cout)
        self.branch1 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=(1, kernel_size),
                             stride=(1, 1),
                             dilation=dilated_num,
                             groups=gcd_value,
                             padding=(0, padding_width)),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )

        kernel_size = kernel_sizes[1]
        dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
        padding_width = (dilated_kernel_size - 1) // 2

        self.branch2 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=(1, kernel_size),
                             stride=(1, 1),
                             dilation=dilated_num,
                             groups=gcd_value,
                             padding=(0, padding_width)),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
        kernel_size = kernel_sizes[2]
        dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
        padding_width = (dilated_kernel_size - 1) // 2
        self.branch3 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=(1, kernel_size),
                             stride=(1, 1),
                             dilation=dilated_num,
                             groups=gcd_value,
                             padding=(0, padding_width)),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )

        kernel_size = kernel_sizes[3]
        dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
        padding_width = (dilated_kernel_size - 1) // 2
        self.branch4 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=(1, kernel_size),
                             stride=(1, 1),
                             dilation=dilated_num,
                             groups=gcd_value,
                             padding=(0, padding_width)),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 拼接所有分支的输出
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  # 在通道维度上拼接

class Inception_block(nn.Module):
    def __init__(self, C_in, C_out,kernel_sizes,dilated_num):
        super(Inception_block, self).__init__()

        if C_out % 4 != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by 4.")

        sub_Cout = int(C_out / 4)

        kernel_size = kernel_sizes[0]
        dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
        padding_width = (dilated_kernel_size - 1) // 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=(1, kernel_size),
                             stride=(1, 1),
                             dilation=dilated_num,
                             padding=(0, padding_width)),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
        kernel_size = kernel_sizes[1]
        dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
        padding_width = (dilated_kernel_size - 1) // 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=(1, kernel_size),
                             stride=(1, 1),
                             dilation=dilated_num,
                             padding=(0, padding_width)),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
        kernel_size = kernel_sizes[2]
        dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
        padding_width = (dilated_kernel_size - 1) // 2
        self.branch3 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=(1, kernel_size),
                             stride=(1, 1),
                             dilation=dilated_num,
                             padding=(0, padding_width)),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
        kernel_size = kernel_sizes[3]
        dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
        padding_width = (dilated_kernel_size - 1) // 2
        self.branch4 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=(1, kernel_size),
                             stride=(1, 1),
                             dilation=dilated_num,
                             padding=(0, padding_width)),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 拼接所有分支的输出
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  # 在通道维度上拼接

class Inception_ghost(nn.Module):
    def __init__(self, C_in, C_out,kernel_sizes,dilated_num):
        super(Inception_ghost, self).__init__()

        if C_out % 4 != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by 4.")
        sub_Cout = int(C_out / 4)
        kernel_size = kernel_sizes[0]
        self.branch1 = nn.Sequential(
            GhostModule(inp=C_in, oup=sub_Cout, depth_wise_size=kernel_size, dilated_num=dilated_num),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
        kernel_size = kernel_sizes[1]
        self.branch2 = nn.Sequential(
            GhostModule(inp=C_in, oup=sub_Cout, depth_wise_size=kernel_size, dilated_num=dilated_num),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
        kernel_size = kernel_sizes[2]
        self.branch3 = nn.Sequential(
            GhostModule(inp=C_in, oup=sub_Cout, depth_wise_size=kernel_size, dilated_num=dilated_num),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
        kernel_size = kernel_sizes[3]
        self.branch4 = nn.Sequential(
            GhostModule(inp=C_in, oup=sub_Cout, depth_wise_size=kernel_size, dilated_num=dilated_num),
            nn.BatchNorm2d(sub_Cout),
            nn.LeakyReLU()
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 拼接所有分支的输出
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  # 在通道维度上拼接

def incetion_test():
    input_tensor = torch.randn(1, 4, 1, 128)
    # 创建 GhostModule 实例

    # inception_module = Inception_block(C_in=4, C_out=4,kernel_sizes=[1,3,5,7],dilated_num=1 )
    # inception_module = Inception_ghost(C_in=4, C_out=4, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
    inception_module = Inception_group(C_in=4, C_out=4, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
    # 进行前向传播
    output_tensor = inception_module(input_tensor)
    # 打印输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

class conv_mixer_block(nn.Module):
    def __init__(self,input_channel,conv_mode,kernel_size,dilated_num):
        super(conv_mixer_block, self).__init__()
        self.in_dim = input_channel

        if conv_mode == 'norm':
            # 计算扩张后的卷积核大小
            dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
            # 计算padding，确保输出宽度与输入宽度相同
            padding_width = (dilated_kernel_size - 1) // 2
            # 创建卷积层，设置合适的padding
            self.conv1 = nn.Conv2d(input_channel, input_channel,
                                 kernel_size=(1, kernel_size),
                                 stride=(1, 1),
                                 dilation=dilated_num,
                                 groups=input_channel,
                                 padding=(0, padding_width))
        # elif conv_mode == 'ghost':
        #     self.conv1 = GhostModule(inp = input_channel, oup=input_channel,depth_wise_size = kernel_size,dilated_num = dilated_num)
        elif conv_mode == 'inception':
            # 一定要输入多个kernel size 大小
            self.conv1 = Inception_group(C_in = input_channel, C_out = input_channel,kernel_sizes = kernel_size,dilated_num = dilated_num)

        self.BN1 = nn.BatchNorm2d(input_channel)
        self.gelu1 = nn.GELU()

        self.conv2 =nn.Conv2d(input_channel, input_channel,kernel_size=1)
        self.BN2 = nn.BatchNorm2d(input_channel)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.BN1(out1)
        out1 = self.gelu1(out1)
        out1 = out1 + x
        out2 = self.conv2(out1)
        out2 = self.BN2(out2)
        out2 = self.gelu2(out2)
        return out2

def conv_test():
    # 创建输入数据
    data = torch.randn(10, 15, 1, 128)  # 10个样本，1个通道，高度2，宽度128
    # 设置卷积核大小
    kernel_size = 11
    dilated_num = 2
    # 计算扩张后的卷积核大小
    dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
    # 计算padding，确保输出宽度与输入宽度相同
    padding_width = (dilated_kernel_size - 1) // 2
    # 创建卷积层，设置合适的padding
    gcd_value = math.gcd(15, 5)
    conv = nn.Conv2d(15, 5, kernel_size=(1, kernel_size), stride=(1, 1), dilation=dilated_num,
                     groups=gcd_value,
                     padding=(0, padding_width))
    # 执行卷积操作
    result = conv(data)
    # 输出结果的形状
    print(result.shape)



def conv_mixer_test():

    input_tensor = torch.randn(16, 16, 1, 128)
    # 创建 GhostModule 实例 norm  inception
    # inception_module = conv_mixer_block(input_channel=16,conv_mode='inception',kernel_size=[1,3,5,7],dilated_num=1 )
    conv_mixer = conv_mixer_block(input_channel=16, conv_mode='norm', kernel_size=3,dilated_num=1)
    # 进行前向传播
    output_tensor = conv_mixer(input_tensor)
    # 打印输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")


### 下面分形网络的通道输出至少是可以除以16


class Fractal_inception(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Fractal_inception, self).__init__()
        if output_channel % 4 != 0:
            raise ValueError(f"C_out ({output_channel}) must be divisible by 4.")

        sub_Cout = int(output_channel / 4)

        self.conv0001 = Inception_ghost(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0002 = Inception_ghost(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0010 = Inception_ghost(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0100 = Inception_ghost(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv1000 = Inception_ghost(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

        self.conv0003 = Inception_ghost(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0004 = Inception_ghost(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0020 = Inception_ghost(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

        self.conv0005 = Inception_ghost(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0006 = Inception_ghost(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0030 = Inception_ghost(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0007 = Inception_ghost(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0008 = Inception_ghost(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0040 = Inception_ghost(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0200 = Inception_ghost(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

    def forward(self, x):
        ######################
        right1out = self.conv0002(self.conv0001(x))
        right2out = self.conv0010(x)
        rightout1 = torch.cat([right2out,right1out], 1)  # 在通道维度上拼接
        ######################
        right3out = self.conv0004(self.conv0003(rightout1))
        right4out = self.conv0020(rightout1)
        mid_out1   = self.conv0100(x)
        rightout2 = torch.cat([mid_out1,right4out, right3out], 1)  # 在通道维度上拼接
        #########################
        right5out = self.conv0006(self.conv0005(rightout2))
        right6out = self.conv0030(rightout2)
        rightout3 = torch.cat([right6out, right5out], 1)  # 在通道维度上拼接
        ######################
        right7out = self.conv0008(self.conv0007(rightout3))
        right8out = self.conv0040(rightout3)
        mid_out2   = self.conv0200(rightout2)
        left_out = self.conv1000(x)
        result = torch.cat([left_out,mid_out2,right8out, right7out], 1)  # 在通道维度上拼接

        return result

def Fractal_incep_test():
    input_tensor = torch.randn(16, 16, 1, 128)
    # 创建 GhostModule 实例 norm  inception
    # inception_module = conv_mixer_block(input_channel=16,conv_mode='inception',kernel_size=[1,3,5,7],dilated_num=1 )
    Fractal_incep_exm = Fractal_inception(input_channel=16, output_channel=16)
    # 进行前向传播
    output_tensor = Fractal_incep_exm(input_tensor)
    # 打印输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")


class Conv_DownSampling(nn.Module):
    def __init__(self, C):
        super(Conv_DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(C),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

class Dila_DownSampling(nn.Module):
    def __init__(self, C):
        super(Dila_DownSampling, self).__init__()
        dila_num = 2
        self.Down = nn.Sequential(
            # 使用扩张卷积进行下采样
            nn.Conv2d(C, C, kernel_size=(1, 3), stride=(1, 2), padding=(0, dila_num), dilation=(1,dila_num)),
            nn.BatchNorm2d(C),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

class Avg_DownSampling(nn.Module):
    def __init__(self, C):
        super(Avg_DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用平均池化进行下采样，stride=2表示宽高各减半
            nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(C),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

def DownSamp_test():
    input_tensor = torch.randn(16, 16, 1, 128)
    # 创建 GhostModule 实例 norm  inception
    # inception_module = conv_mixer_block(input_channel=16,conv_mode='inception',kernel_size=[1,3,5,7],dilated_num=1 )
    # DownSamp_exm = Conv_DownSampling(16)
    # DownSamp_exm = Dila_DownSampling(16)
    DownSamp_exm = Avg_DownSampling(16)
    # 进行前向传播
    output_tensor = DownSamp_exm(input_tensor)
    # 打印输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")


class depth_conv_mixer(nn.Module):
    def __init__(self,input_channel,conv_mode,kernel_list,dilated_list):
        super(depth_conv_mixer, self).__init__()
        self.mixer1 = conv_mixer_block(input_channel, conv_mode, kernel_list, dilated_list)
        self.mixer2 = conv_mixer_block(input_channel, conv_mode, kernel_list, dilated_list)
        self.mixer3 = conv_mixer_block(input_channel, conv_mode, kernel_list, dilated_list)
        self.mixer4 = conv_mixer_block(input_channel, conv_mode, kernel_list, dilated_list)
        self.mixer5 = conv_mixer_block(input_channel, conv_mode, kernel_list, dilated_list)



    #
    #
    # def forward(self, x):


if __name__ == '__main__':
    incetion_test()
    # conv_test()
    # conv_mixer_test()
    # ghost_test()
    # Fractal_incep_test()
    # DownSamp_test()
    # test_PatchEmbedding()
