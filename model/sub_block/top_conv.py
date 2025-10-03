import torch
import torch.nn as nn
from sympy.strategies.core import switch
from transformers.models.rwkv.modeling_rwkv import rwkv_linear_attention

from model.sub_block.low_conv import BasicNormConv,PyramidConvCompress,dw_decompos_conv,Swish_act
from model.sub_block.mid_conv import Inception_group_cat,inception_group_sum,inception_ghost_sum,inception_sum,Inception_cat,Inception_ghost_cat
from model.sub_block.mid_conv import res_incep,channel_shuffle
from torch.nn import functional as F
import time


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
    print("SK_Channel_atten2D_test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试参数
    img_size = 256
    C_in = 4
    C_out = 64
    kernel_sizes = [3, 5, 7, 9]
    dilated_list = [1, 1, 1, 1]

    # 清空GPU缓存并记录初始显存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
    print(f"初始显存占用: {initial_memory:.2f} MB")

    # 创建输入张量
    x = torch.randn(16, C_in, img_size, img_size).to(device)
    input_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory
    print(f"输入张量显存占用: {input_memory:.2f} MB")

    # 创建模型
    model = SK_Channel_atten2D(img_size, C_in, C_out, kernel_sizes, dilated_list).to(device)
    model_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory - input_memory
    print(f"模型参数显存占用: {model_memory:.2f} MB")

    # 前向传播
    start_time = time.time()
    output = model(x)
    forward_time = time.time() - start_time
    print(f"前向传播时间: {forward_time:.4f} 秒")

    forward_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory - input_memory - model_memory
    print(f"前向传播中间变量显存占用: {forward_memory:.2f} MB")

    # 统计信息
    total_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
    print(f"总显存占用: {total_memory:.2f} MB")

    # 峰值显存使用
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f"峰值显存使用: {peak_memory:.2f} MB")

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    return output

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

class fractal_conv(nn.Module):
    def __init__(self,C_in,C_out,kernel_list,dilated_list,inception_module):
        super(fractal_conv, self).__init__()
        if C_out % 4 != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by 4")
        norm_flag = True
        print("fractal_conv norm:",norm_flag)
        sub_Cout = int(C_out / 4)

        self.conv0001 = inception_module(C_in=C_in, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0002 = inception_module(C_in=sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0010 = inception_module(C_in=C_in, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0100 = inception_module(C_in=C_in, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv1000 = inception_module(C_in=C_in, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)

        self.conv0003 = inception_module(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0004 = inception_module(C_in=sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0020 = inception_module(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)

        self.conv0005 = inception_module(C_in=sub_Cout*3, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0006 = inception_module(C_in=sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0030 = inception_module(C_in=sub_Cout*3, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0007 = inception_module(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0008 = inception_module(C_in=sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0040 = inception_module(C_in=sub_Cout+sub_Cout, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)
        self.conv0200 = inception_module(C_in=sub_Cout*3, C_out=sub_Cout, kernel_list=kernel_list, dilated_list=dilated_list, norm = norm_flag)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 示例参数
    batch_size = 16
    channels = 512
    img_H = 16
    img_W = 16
    kernel_list = [3, 5, 7, 9]
    dilated_list = [1, 1, 1, 1]
    C_out = 512

    # 创建测试输入
    x = torch.randn(batch_size, channels, img_H, img_W).to(device)

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
    print("-" * 80)

    # 循环测试每个模块
    for i, (module_class, module_name) in enumerate(zip(inception_modules, module_names)):
        try:
            print(f"测试 {i + 1}/{len(inception_modules)}: {module_name}")

            # 清空GPU缓存并记录初始显存
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
            input_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory
            print(f"  输入张量显存占用: {input_memory:.2f} MB")

            # 创建残差块
            residual_block = fractal_conv(
                C_in=channels,
                C_out=C_out,
                kernel_list=kernel_list,
                dilated_list=dilated_list,
                inception_module=module_class
            ).to(device)

            model_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory - input_memory
            print(f"  模型参数显存占用: {model_memory:.2f} MB")

            # 前向传播
            start_time = time.time()
            output_tensor = residual_block(x)
            forward_time = time.time() - start_time

            forward_memory = torch.cuda.memory_allocated(
                device) / 1024 ** 2 - initial_memory - input_memory - model_memory
            print(f"  前向传播时间: {forward_time:.4f} 秒")
            print(f"  前向传播中间变量显存占用: {forward_memory:.2f} MB")

            # 峰值显存使用
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print(f"  峰值显存使用: {peak_memory:.2f} MB")

            # 打印结果
            print(f"  输出形状: {output_tensor.shape}")
            print(f"  测试通过 ✓")
            print("-" * 50)

        except Exception as e:
            print(f"测试 {module_name} 时出错: {e}")
            print("-" * 50)


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

# CA(X) = X+γ ⊙ ( X - GELU(XW) )
# Y= GELU(DW:x:(Conv1x1(Norm(X)))
# Z=Conv1x1(CA(Y))+ X.
class subtract_Channel_atten(nn.Module):
    def __init__(self, C_in,gama = 2):
        super(subtract_Channel_atten,self).__init__()
        C_out = gama*C_in
        self.conv_begin =nn.Sequential(dw_decompos_conv(inp = C_in,oup = C_out, dw_kernel = 3, dw_dilated = 1),
                                   nn.BatchNorm2d(C_out),
                                   nn.GELU())
        self.CA_conv = BasicNormConv(C_in = C_out, C_out = 1, kernel_size = 1,gelu = True,norm = False)
        self.conv_end = BasicNormConv(C_in = C_out, C_out = C_in, kernel_size = 1,gelu=False,norm = False)
        self.sigma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        CA_in = self.conv_begin(x)
        CA_out =CA_in +self.sigma *  (CA_in - self.CA_conv(CA_in))
        out = self.conv_end(CA_out)
        return out

class MoGA_dw_conv(nn.Module):
    def __init__(self, C_in, ratio_list, kernel_list, dilated_list):
        super(MoGA_dw_conv, self).__init__()
        self.conv_begin = BasicNormConv(C_in=C_in, C_out=C_in, kernel_size=5, dilation=1, groups=C_in,gelu=False,norm = False)
        self.C_mid_list = [int(C_in * ratio) for ratio in ratio_list]

        assert sum(self.C_mid_list) == C_in, "Sum of mid channels must equal C_in"
        self.conv_list = nn.ModuleList()

        for i in range(len(self.C_mid_list)):
            if kernel_list[i] == 0:
                # 使用恒等映射层
                self.conv_list.append(nn.Identity())
            else:
                self.conv_list.append(BasicNormConv(
                    C_in=self.C_mid_list[i],
                    C_out=self.C_mid_list[i],
                    kernel_size=kernel_list[i],
                    dilation=dilated_list[i],
                    groups=self.C_mid_list[i],
                    gelu=False,
                    norm=False
                ))
        self.conv_end = BasicNormConv(C_in=C_in, C_out=C_in, kernel_size=1, gelu = False, norm = False)
        self.silu = Swish_act()

    def forward(self, x):

        out1 = channel_shuffle(self.conv_begin(x))
        out_mid_list = []
        index_sum = 0

        for i in range(len(self.conv_list)):

            chunk = out1[:, index_sum:index_sum + self.C_mid_list[i], :, :]
            out_mid_list.append(self.conv_list[i](chunk))
            index_sum += self.C_mid_list[i]

        out_end = self.silu(self.conv_end(torch.cat(out_mid_list, dim=1)))

        return out_end


#Z=X+Moga(FD(Norm(X)))
#Y = Conv1X1(X)
#Z=GELU(Y+gama⊙(Y-GAP(Y)))
class MoGA_atten(nn.Module):
    def __init__(self, C_in,ratio_list,kernel_list,dilated_list):
        super(MoGA_atten,self).__init__()


        self.norm_layer = nn.BatchNorm2d(C_in)
        self.conv1X1_begin =BasicNormConv(C_in=C_in, C_out=C_in, kernel_size=1, gelu = False, norm = False)
        self.gelu = nn.GELU()
        self.silu = Swish_act()

        self.moga_res_conv1X1 = BasicNormConv(C_in = C_in, C_out = C_in, kernel_size = 1, gelu = False, norm = False)
        self.moga_out_conv1X1 = BasicNormConv(C_in=C_in, C_out=C_in, kernel_size=1, gelu = False, norm = False)
        self.moga_dwConv =MoGA_dw_conv(C_in = C_in, ratio_list = ratio_list,kernel_list = kernel_list, dilated_list= dilated_list)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out_begin = self.conv1X1_begin(self.norm_layer(x))
        out_gate = self.gelu(out_begin + self.sigma*(out_begin - F.adaptive_avg_pool2d(out_begin, output_size=1)))
        out_moga = self.moga_out_conv1X1( self.silu(self.moga_res_conv1X1(out_gate)) + self.moga_dwConv(out_gate) )
        out_moga = out_moga + x
        return out_moga


class MSAA_space_atten(nn.Module):
    def __init__(self, C_in,space_pool_kernel,kernel_list,dilated_list):
        super(MSAA_space_atten,self).__init__()
        C_space = C_in
        self.space_conv_begin = BasicNormConv(C_in=C_in, C_out=C_space, kernel_size=1,gelu=False,norm = False)
        self.space_multi_scale = inception_sum(C_in = C_space, C_out = C_space,kernel_list = kernel_list,dilated_list = dilated_list,gelu=False,norm = False)

        self.space_pool_conv = nn.Sequential(BasicNormConv(C_in=C_space, C_out=C_space//2, kernel_size=1,gelu=False,norm = False),
                                             BasicNormConv(C_in=C_space//2, C_out=1, kernel_size=1,gelu=False,norm = False),
                                             BasicNormConv(C_in=1, C_out=1, kernel_size=space_pool_kernel,gelu=False,norm = False),
                                             nn.Sigmoid())
        self.space_conv_end = BasicNormConv(C_in=C_space, C_out=C_in, kernel_size=1)

    def forward(self, x):
        out_mid = self.space_multi_scale(self.space_conv_begin(x))
        out = self.space_conv_end(self.space_pool_conv(out_mid)* out_mid)
        return out


class MSAA_channel_atten(nn.Module):

    def __init__(self, img_size,C_in):
        super(MSAA_channel_atten,self).__init__()
        self.compress = PyramidConvCompress(img_size = img_size, C_in = C_in, down_sample_size = 8)
        self.conv_end = nn.Sequential(BasicNormConv(C_in = C_in, C_out = C_in, kernel_size = 1,gelu=True,norm = True),
                                      BasicNormConv(C_in=C_in, C_out=C_in, kernel_size=1,gelu=True,norm = True) )

    def forward(self, x):

        return self.conv_end( self.compress(x))


class MSAA_origin(nn.Module):
    def __init__(self, C_in,img_size, kernel_list, dilated_list):
        super(MSAA_origin, self).__init__()
        self.space_atten =  MSAA_space_atten(C_in=C_in,
                                             space_pool_kernel=7,
                                             kernel_list=kernel_list,
                                             dilated_list=dilated_list)
        self.channel_atten = MSAA_channel_atten(img_size = img_size, C_in = C_in)

    def forward(self, x):

        return x + self.channel_atten(x)*self.space_atten(x)


class MSAA_space_channel(nn.Module):
    def __init__(self, C_in,img_size, kernel_list, dilated_list):
        super(MSAA_space_channel, self).__init__()
        self.space_atten =  MSAA_space_atten(C_in=C_in,
                                             space_pool_kernel=7,
                                             kernel_list=kernel_list,
                                             dilated_list=dilated_list)
        self.channel_atten = MSAA_channel_atten(img_size = img_size, C_in = C_in)

    def forward(self, x):
        out_space = self.space_atten(x) + x
        out_channel = self.channel_atten(out_space)*out_space
        return out_channel

def MSAA_space_atten_test():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    channels = 64
    img_H = 64
    img_W = 64

    # 测试包含恒等层的情况（kernel_list中包含0）

    kernel_list = [3, 5, 7]  # 包含恒等层
    dilated_list = [1, 1, 1]

    x = torch.randn(batch_size, channels, img_H, img_W).to(device)
    model = MSAA_space_atten(
        C_in=channels,
        space_pool_kernel=7,
        kernel_list=kernel_list,
        dilated_list=dilated_list
    ).to(device)

    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == x.shape, "Output shape should match input shape"

class MSAA_channel_space(nn.Module):
    def __init__(self, C_in,img_size, kernel_list, dilated_list):
        super(MSAA_channel_space, self).__init__()
        self.space_atten =  MSAA_space_atten(C_in=C_in,
                                             space_pool_kernel=7,
                                             kernel_list=kernel_list,
                                             dilated_list=dilated_list)
        self.channel_atten = MSAA_channel_atten(img_size = img_size, C_in = C_in)

    def forward(self, x):
        out_channel = self.channel_atten(x) * x
        out_space = self.space_atten(out_channel) + out_channel

        return out_space


if __name__ == '__main__':
    # SK_Channel_atten2D_test()
    print(1)
    fractal_conv_test()