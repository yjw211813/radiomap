import torch
import torch.nn as nn
from model.sub_block.low_conv import BasicNormConv,PyramidConvCompress
from model.sub_block.mid_conv import Inception_group_cat,inception_group_sum,inception_ghost_sum,inception_sum,Inception_cat,Inception_ghost_cat
from model.sub_block.mid_conv import res_incep,channel_shuffle
'''
    高层次模块
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


class conv_mixer(nn.Module):
    def __init__(self,input_channel,conv_mode,kernel_size,dilate):
        super(conv_mixer, self).__init__()
        self.in_dim = input_channel

        if conv_mode == 'norm':
            self.conv1 = BasicNormConv(C_in=input_channel, C_out=input_channel, kernel_size=kernel_size, dilation=dilate,groups=input_channel)
        elif conv_mode == 'inception_cat':
            self.conv1 = Inception_group_cat(C_in=input_channel, C_out=input_channel, kernel_list=kernel_size, dilated_list=dilate)
        elif conv_mode == 'inception_sum':
            self.conv1 = inception_group_sum(C_in=input_channel, C_out=input_channel, kernel_list=kernel_size, dilated_list=dilate)

        self.BN1 = nn.BatchNorm2d(input_channel)
        self.gelu1 = nn.GELU()

        self.conv2 =nn.Conv2d(input_channel, input_channel,kernel_size=1)
        self.BN2 = nn.BatchNorm2d(input_channel)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        out1 = self.gelu1(self.BN1(self.conv1(x)))
        out1 = out1 + x
        out2 = self.gelu2(self.BN2(self.conv2(out1)))
        return out2


def conv_mixer_test():

        # 示例参数
    batch_size = 10  # 批次大小
    channels = 20  # 通道数
    img_H = 128  # 序列长度
    img_W = 128  # 第三维度长度

    kernel_list = [3,5,7,9,11]  
    dilated_list = [1,2,4,8,4]
    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据
    print(f"Input shape: {x.shape}")
    inception_module = conv_mixer(input_channel = channels,conv_mode = "inception_cat",kernel_size = kernel_list,dilate = dilated_list)
    output_tensor = inception_module(x)
    
    print(f"inception_cat test Output shape: {output_tensor.shape}")

    inception_module = conv_mixer(input_channel = channels,conv_mode = "inception_sum",kernel_size = kernel_list,dilate = dilated_list)
    output_tensor = inception_module(x)
    
    print(f"inception_sum test Output shape: {output_tensor.shape}")

    inception_module = conv_mixer(input_channel = channels,conv_mode = "norm",kernel_size = 3,dilate = 1)
    output_tensor = inception_module(x)
    
    print(f"norm test Output shape: {output_tensor.shape}")






### 下面分形网络的通道输出至少是可以除以16

class fractal_conv(nn.Module):
    def __init__(self,C_in,C_out,kernel_list,dilated_list,inception_module):
        super(fractal_conv, self).__init__()
        if C_out % 4 != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by 4")


        sub_Cout = int(C_out / 4)

        self.conv0001 = inception_module(C_in=C_in, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0002 = inception_module(C_in=sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0010 = inception_module(C_in=C_in, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0100 = inception_module(C_in=C_in, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv1000 = inception_module(C_in=C_in, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)

        self.conv0003 = inception_module(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0004 = inception_module(C_in=sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0020 = inception_module(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)

        self.conv0005 = inception_module(C_in=sub_Cout*3, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0006 = inception_module(C_in=sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0030 = inception_module(C_in=sub_Cout*3, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0007 = inception_module(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0008 = inception_module(C_in=sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0040 = inception_module(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)
        self.conv0200 = inception_module(C_in=sub_Cout*3, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list)

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

def fractal_conv_test():
    # 示例参数
    batch_size = 2
    channels = 32
    img_H = 64
    img_W = 64
    kernel_list = [ 3, 5, 7, 9]
    dilated_list = [ 2, 4, 8, 4]
    C_out = 64

    # 创建测试输入
    x = torch.randn(batch_size, channels, img_H, img_W)

    # 要测试的模块列表
    inception_modules = [
        Inception_group_cat,
        inception_sum,
        inception_ghost_sum,
        Inception_cat,
        Inception_ghost_cat
    ]

    # 模块名称列表（用于输出）
    module_names = [
        "Inception_group_cat",
        "inception_sum",
        "inception_ghost_sum",
        "Inception_cat",
        "Inception_ghost_cat"
    ]

    print("开始测试各种 Inception 模块...")
    print(f"输入形状: {x.shape}")
    print("-" * 50)

    # 循环测试每个模块
    for i, (module_class, module_name) in enumerate(zip(inception_modules, module_names)):
        try:
            print(f"测试 {i + 1}/{len(inception_modules)}: {module_name}")

            # 创建残差块
            residual_block = fractal_conv(
                C_in=channels,
                C_out=C_out,
                kernel_list=kernel_list,
                dilated_list=dilated_list,
                inception_module=module_class
            )

            # 前向传播
            output_tensor = residual_block(x)

            # 打印结果
            print(f"输出形状: {output_tensor.shape}")
            print(f"测试通过 ✓")
            print("-" * 30)

        except Exception as e:
            print(f"测试 {module_name} 时出错: {e}")
            print("-" * 30)

class res_incep_ghost_sum(res_incep):
    def __init__(self, C_in, C_out, kernel_list, dilated_list):
        super().__init__(C_in, C_out, kernel_list, dilated_list, inception_ghost_sum)

class res_Inception_group_cat(res_incep):
    def __init__(self, C_in, C_out, kernel_list, dilated_list):
        super().__init__(C_in, C_out, kernel_list, dilated_list, Inception_group_cat)

class res_inception_sum(res_incep):
    def __init__(self, C_in, C_out, kernel_list, dilated_list):
        super().__init__(C_in, C_out, kernel_list, dilated_list, inception_sum)

class res_inception_group_sum(res_incep):
    def __init__(self, C_in, C_out, kernel_list, dilated_list):
        super().__init__(C_in, C_out, kernel_list, dilated_list, inception_group_sum)

class res_Inception_cat(res_incep):
    def __init__(self, C_in, C_out, kernel_list, dilated_list):
        super().__init__(C_in, C_out, kernel_list, dilated_list, Inception_cat)

class res_Inception_ghost_cat(res_incep):
    def __init__(self, C_in, C_out, kernel_list, dilated_list):
        super().__init__(C_in, C_out, kernel_list, dilated_list, Inception_ghost_cat)

def res_Inception_ghost_cat_test():
        # 示例参数
    batch_size = 2
    channels = 32
    img_H = 64
    img_W = 64
    kernel_list = [ 3, 5, 7, 9]
    dilated_list = [ 2, 4, 8, 4]
    C_out = 64

    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据

    inception_module = res_Inception_ghost_cat(C_in=channels, C_out=C_out, kernel_list=kernel_list, dilated_list=dilated_list)
    output_tensor = inception_module(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_tensor.shape}")

class res_fractal_conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_list, dilated_list, inception_module):
        super(res_fractal_conv,self).__init__()
        self.frac_conv = fractal_conv(C_in=C_in,
                                      C_out=C_out, 
                                      kernel_list=kernel_list, 
                                      dilated_list=dilated_list,
                                      inception_module = inception_module)
        self.short_path = BasicNormConv(C_in, C_out, kernel_size=1)
    def forward(self, x):
        return channel_shuffle(self.frac_conv(x) + self.short_path(x))

def res_fractal_conv_test():
    batch_size = 2
    channels = 32
    img_H = 64
    img_W = 64
    kernel_list = [ 3, 5, 7, 9]
    dilated_list = [ 2, 4, 8, 4]
    C_out = 64
    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据

    inception_module = res_fractal_conv(C_in=channels,
                                        C_out=C_out, 
                                        kernel_list=kernel_list, 
                                        dilated_list=dilated_list,
                                        inception_module = res_Inception_ghost_cat)
    output_tensor = inception_module(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_tensor.shape}")





# 运行测试
if __name__ == "__main__":

    # SE_Channel_attan2D_test()
    # SK_Channel_atten2D_test()

    # conv_mixer_test()
    # fractal_conv_test()
    res_fractal_conv_test()
    res_Inception_ghost_cat_test()


