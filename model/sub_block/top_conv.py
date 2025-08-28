import torch
import torch.nn as nn
from model.sub_block.low_conv import BasicNormConv,PyramidConvCompress

'''
    常用的通道注意力模块
'''
# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Channel_attan2D(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Channel_attan2D, self).__init__()
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).reshape(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).reshape(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

def SE_Channel_attan2D_test():
    # 测试 SE_Block
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 32, 128, 128).to(device)  # 假设输入是一个 batch_size 为 2，通道数为 32，8x8 的特征图
    se_block = SE_Channel_attan2D(inchannel=32).to(device)
    output = se_block(x)
    print(output.shape)  # 应输出 (2, 32, 128, 128)


# 三入 相加 或者两入相加 得到组合特征
# 这个应该是有个注意力选择机制在其中的 inception
class SK_Channel_atten2D(nn.Module):
    def __init__(self, img_size,C_in,C_out,kernel_sizes,dilated_list):
        super(SK_Channel_atten2D, self).__init__()
        
        self.C_out = C_out
        self.num_branches = len(kernel_sizes)
        ratio = 1
        down_sample_size = 8

        # 定义CNN分支 feature extracture spit
        self.branches = nn.ModuleList([
                    BasicNormConv(C_in, C_out, kernel_sizes[i], dilated_list[i])
                    for i in range(self.num_branches)
                ])

        # 分支融合后压缩模块 使用金字塔卷积压缩
        self.compress_info = PyramidConvCompress(img_size, C_out, down_sample_size)

        # 特征线性变化层
        self.feature_transform = nn.Sequential(
            nn.Linear(C_out, C_out // ratio, bias=False),
            nn.BatchNorm1d(C_out // ratio),
            nn.GELU(),
            nn.Linear(C_out // ratio, self.num_branches * C_out, bias=False),
        )
        self.weight_layer = nn.Softmax(dim=-1)  # 修改为对最后一个维度进行Softmax
    def forward(self, x):
        
        # 1. 通过各个分支处理输入
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))

        # 2. 将分支输出相加得到融合特征
        fused = torch.stack(branch_outputs, dim=0).sum(dim=0)
        # 3. 通过压缩模块降维
        fused = self.compress_info(fused)
        compressed = fused.reshape(x.size(0), self.C_out)  # 展平为(batch_size, C_out)
       
       # 4. 通过全连接层生成权重
        weights = self.feature_transform(compressed)
        # 5. 拆分成多个分支的权重
        weights_split = weights.reshape(x.size(0), self.num_branches, self.C_out)

        weights_split = self.weight_layer(weights_split)
        # 形状变为(b, num_branches, C_out, 1, 1)
        weights_split = weights_split.unsqueeze(-1).unsqueeze(-1)

        for i in range(self.num_branches):
            if i == 0:
                # 应用对应分支的权重
                final_output = branch_outputs[i] * weights_split[:, i, :, :, :]
            else:
                final_output += branch_outputs[i] * weights_split[:, i, :, :, :]
        return final_output

def SK_Channel_atten2D_test():
    # 测试参数
    img_size = 256
    C_in = 4
    C_out = 8
    kernel_sizes = [3, 5, 7 ,9]
    dilated_list = [1, 1, 1 ,1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建测试输入和模型
    x = torch.randn(2, C_in, img_size, img_size).to(device)
    model = SK_Channel_atten2D(img_size, C_in, C_out, kernel_sizes, dilated_list).to(device)
    
    # 前向传播
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)  # 应输出 (2, 64, 128, 128)


class conv_mixer_block2D(nn.Module):
    def __init__(self,input_channel,conv_mode,kernel_size,dilated_num):
        super(conv_mixer_block2D, self).__init__()
        self.in_dim = input_channel

        if conv_mode == 'norm':
            # 计算扩张后的卷积核大小
            dilated_kernel_size = (kernel_size - 1) * dilated_num + 1
            # 计算padding，确保输出宽度与输入宽度相同
            padding_width = (dilated_kernel_size - 1) // 2
            # 创建卷积层，设置合适的padding
            self.conv1 = nn.Conv2d(input_channel, input_channel,
                                 kernel_size=kernel_size,
                                 stride=(1, 1),
                                 dilation=dilated_num,
                                 groups=input_channel,
                                 padding=padding_width)
        # elif conv_mode == 'ghost':
        #     self.conv1 = GhostModule(inp = input_channel, oup=input_channel,depth_wise_size = kernel_size,dilated_num = dilated_num)
        elif conv_mode == 'inception':
            # 一定要输入多个kernel size 大小
            self.conv1 = Inception_group2D(C_in = input_channel, C_out = input_channel,kernel_sizes = kernel_size,dilated_num = dilated_num)

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

    input_tensor = torch.randn(4, 16, 128, 128)
    # 创建 GhostModule 实例 norm  inception
    # inception_module = conv_mixer_block(input_channel=16,conv_mode='inception',kernel_size=[1,3,5,7],dilated_num=1 )
    conv_mixer = conv_mixer_block2D(input_channel=16, conv_mode='norm', kernel_size=3,dilated_num=1)
    # 进行前向传播
    output_tensor = conv_mixer(input_tensor)
    # 打印输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

### 下面分形网络的通道输出至少是可以除以16

class Fractal_inception2D(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Fractal_inception2D, self).__init__()
        if output_channel % 4 != 0:
            raise ValueError(f"C_out ({output_channel}) must be divisible by 4.")

        sub_Cout = int(output_channel / 4)

        self.conv0001 = Inception_ghost2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0002 = Inception_ghost2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0010 = Inception_ghost2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0100 = Inception_ghost2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv1000 = Inception_ghost2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

        self.conv0003 = Inception_ghost2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0004 = Inception_ghost2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0020 = Inception_ghost2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

        self.conv0005 = Inception_ghost2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0006 = Inception_ghost2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0030 = Inception_ghost2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0007 = Inception_ghost2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0008 = Inception_ghost2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0040 = Inception_ghost2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0200 = Inception_ghost2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

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
    input_tensor = torch.randn(16, 16, 128, 128)
    # 创建 GhostModule 实例 norm  inception
    Fractal_incep_exm = Fractal_inception2D(input_channel=16, output_channel=16)
    output_tensor = Fractal_incep_exm(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

class Res_Fractal_inception2D(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Res_Fractal_inception2D, self).__init__()
        if output_channel % 4 != 0:
            raise ValueError(f"C_out ({output_channel}) must be divisible by 4.")

        sub_Cout = int(output_channel / 4)

        self.conv0001 = Res_Inception_ghost2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0002 = Res_Inception_ghost2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0010 = Res_Inception_ghost2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0100 = Res_Inception_ghost2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv1000 = Res_Inception_ghost2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

        self.conv0003 = Res_Inception_ghost2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0004 = Res_Inception_ghost2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0020 = Res_Inception_ghost2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

        self.conv0005 = Res_Inception_ghost2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0006 = Res_Inception_ghost2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0030 = Res_Inception_ghost2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0007 = Res_Inception_ghost2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0008 = Res_Inception_ghost2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0040 = Res_Inception_ghost2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0200 = Res_Inception_ghost2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

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

def Res_Fractal_incep_test():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(16, 16, 128, 128).to(device)
    # 创建 GhostModule 实例 norm  inception
    Fractal_incep_exm = Res_Fractal_inception2D(input_channel=16, output_channel=16).to(device)
    output_tensor = Fractal_incep_exm(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")


class Fractal_multi_scale2D(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Fractal_multi_scale2D, self).__init__()
        if output_channel % 4 != 0:
            raise ValueError(f"C_out ({output_channel}) must be divisible by 4.")

        sub_Cout = int(output_channel / 4)

        self.conv0001 = multi_scale_block2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0002 = multi_scale_block2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0010 = multi_scale_block2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0100 = multi_scale_block2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv1000 = multi_scale_block2D(C_in=input_channel, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

        self.conv0003 = multi_scale_block2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0004 = multi_scale_block2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0020 = multi_scale_block2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

        self.conv0005 = multi_scale_block2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0006 = multi_scale_block2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0030 = multi_scale_block2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0007 = multi_scale_block2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0008 = multi_scale_block2D(C_in=sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0040 = multi_scale_block2D(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.conv0200 = multi_scale_block2D(C_in=sub_Cout*3, C_out=sub_Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)

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

def Fractal_multi_scale2D_test():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(16, 16, 128, 128).to(device)
    # 创建 GhostModule 实例 norm  inception
    Fractal_incep_exm = Fractal_multi_scale2D(input_channel=16, output_channel=16).to(device)
    output_tensor = Fractal_incep_exm(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")








# 运行测试
if __name__ == "__main__":
    # SE_Channel_attan2D_test()
    SK_Channel_atten2D_test()















# # 全局平均池化+全局最大池化 组合而成的通道注意力
# class Ave_Max_CA2D(nn.Module):
#     def __init__(self, inchannel, ratio=16):
#         super(Ave_Max_CA2D, self).__init__()
#         # 全局平均池化(Fsq操作)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         # 两个全连接层(Fex操作)
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
#             nn.GELU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # 读取批数据图片数量及通道数
#         b, c, h, w = x.size()
#         # Fsq操作：经池化后输出b*c的矩阵
#         y = self.gap(x).reshape(b, c)
#         # Fex操作：经全连接层输出（b，c，1，1）矩阵
#         y = self.fc(y).reshape(b, c, 1, 1)
#         # Fscale操作：将得到的权重乘以原来的特征图x
#         return x * y.expand_as(x)
#
# def Ave_Max_CA2D_test():
#     # 测试 SE_Block
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(2, 32, 128, 128).to(device)  # 假设输入是一个 batch_size 为 2，通道数为 32，8x8 的特征图
#     se_block = SE_Channel_attan2D(inchannel=32).to(device)
#     output = se_block(x)
#     print(output.shape)  # 应输出 (2, 32, 128, 128)
#
# # max 空间注意力
#
# class Max_SA2D(nn.Module):
#     def __init__(self, inchannel, ratio=16):
#         super(Max_SA2D, self).__init__()
#         # 全局平均池化(Fsq操作)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         # 两个全连接层(Fex操作)
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
#             nn.GELU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # 读取批数据图片数量及通道数
#         b, c, h, w = x.size()
#         # Fsq操作：经池化后输出b*c的矩阵
#         y = self.gap(x).reshape(b, c)
#         # Fex操作：经全连接层输出（b，c，1，1）矩阵
#         y = self.fc(y).reshape(b, c, 1, 1)
#         # Fscale操作：将得到的权重乘以原来的特征图x
#         return x * y.expand_as(x)
#
# def Ave_Max_CA2D_test():
#     # 测试 SE_Block
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(2, 32, 128, 128).to(device)  # 假设输入是一个 batch_size 为 2，通道数为 32，8x8 的特征图
#     se_block = SE_Channel_attan2D(inchannel=32).to(device)
#     output = se_block(x)
#     print(output.shape)  # 应输出 (2, 32, 128, 128)
#
# # 金子塔池化通道注意力
# class Ave_Max_CA2D(nn.Module):
#     def __init__(self, inchannel, ratio=16):
#         super(Ave_Max_CA2D, self).__init__()
#         # 全局平均池化(Fsq操作)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         # 两个全连接层(Fex操作)
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
#             nn.GELU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # 读取批数据图片数量及通道数
#         b, c, h, w = x.size()
#         # Fsq操作：经池化后输出b*c的矩阵
#         y = self.gap(x).reshape(b, c)
#         # Fex操作：经全连接层输出（b，c，1，1）矩阵
#         y = self.fc(y).reshape(b, c, 1, 1)
#         # Fscale操作：将得到的权重乘以原来的特征图x
#         return x * y.expand_as(x)
#
# def Ave_Max_CA2D_test():
#     # 测试 SE_Block
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(2, 32, 128, 128).to(device)  # 假设输入是一个 batch_size 为 2，通道数为 32，8x8 的特征图
#     se_block = SE_Channel_attan2D(inchannel=32).to(device)
#     output = se_block(x)
#     print(output.shape)  # 应输出 (2, 32, 128, 128)
#
# # 卷积下采样通道注意力
# class Ave_Max_CA2D(nn.Module):
#     def __init__(self, inchannel, ratio=16):
#         super(Ave_Max_CA2D, self).__init__()
#         # 全局平均池化(Fsq操作)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         # 两个全连接层(Fex操作)
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
#             nn.GELU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # 读取批数据图片数量及通道数
#         b, c, h, w = x.size()
#         # Fsq操作：经池化后输出b*c的矩阵
#         y = self.gap(x).reshape(b, c)
#         # Fex操作：经全连接层输出（b，c，1，1）矩阵
#         y = self.fc(y).reshape(b, c, 1, 1)
#         # Fscale操作：将得到的权重乘以原来的特征图x
#         return x * y.expand_as(x)
#
# def Ave_Max_CA2D_test():
#     # 测试 SE_Block
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(2, 32, 128, 128).to(device)  # 假设输入是一个 batch_size 为 2，通道数为 32，8x8 的特征图
#     se_block = SE_Channel_attan2D(inchannel=32).to(device)
#     output = se_block(x)
#     print(output.shape)  # 应输出 (2, 32, 128, 128)
#
# # 坐标注意力？？？
# # 三分支坐标注意力
#
# # qkv 注意力机制