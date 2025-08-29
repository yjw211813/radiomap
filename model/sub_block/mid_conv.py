import torch
import torch.nn as nn
from sympy.strategies.core import switch
import math
from torch.nn import functional as F
from model.sub_block.low_conv import GhostConv2D,BasicNormConv,dw_decompos_conv

'''
    常用的卷积基础模块
'''
def channel_shuffle(x):
    """通道混洗操作"""
    batch_size, num_channels, height, width = x.size()
    # 确保通道数能被分组数整除
    assert num_channels % 4 == 0, "通道数必须能被分组数整除"
    channels_per_group = num_channels // 4
    # 重塑张量以进行分组 - 使用reshape
    x = x.reshape(batch_size, 4, channels_per_group, height, width)
    # 转置分组和通道维度 - 不再需要contiguous()
    x = torch.transpose(x, 1, 2)
    # 重塑回原始尺寸 - 使用reshape
    return x.reshape(batch_size, -1, height, width) 

# 使用多尺度并且进行分组卷积
class Inception_group_cat(nn.Module):
    def __init__(self, C_in, C_out,kernel_list,dilated_list,drop_out=0.05):
        super(Inception_group_cat, self).__init__()

        if C_out % len(kernel_list) != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by len(kernel_list)")


        sub_Cout = int(C_out / len(kernel_list))
        gcd_value = math.gcd(C_in, sub_Cout)
        self.conv_list = nn.ModuleList()
        for i in range(len(kernel_list)):
            self.conv_list.append(BasicNormConv(C_in = C_in, 
                                                C_out = sub_Cout, 
                                                kernel_size = kernel_list[i], 
                                                dilation = dilated_list[i],
                                                groups=gcd_value,
                                                dropout_rate = drop_out))
    def forward(self, x):
        outputs = []
        for conv in self.conv_list:
            outputs.append(conv(x))
        
        # 拼接所有分支的输出
        return torch.cat(outputs, 1)  # 在通道维度上拼接
    

class inception_group_sum(nn.Module):
    def __init__(self, C_in, C_out,kernel_list,dilated_list,drop_out=0.05):
        super(inception_group_sum, self).__init__()


        self.conv_list = nn.ModuleList()
        gcd_value = math.gcd(C_in, C_out)
        for i in range(len(kernel_list)):
            self.conv_list.append(BasicNormConv(C_in = C_in, 
                                                C_out = C_out, 
                                                kernel_size = kernel_list[i], 
                                                dilation = dilated_list[i],
                                                dropout_rate = drop_out,
                                                groups=gcd_value))
    def forward(self, x):
        output = None
        for conv in self.conv_list:
            if output is None:
                output = conv(x)
            else:
                output += conv(x)

        return output  # 在通道维度上拼接



class Inception_cat(nn.Module):
    def __init__(self, C_in, C_out,kernel_list,dilated_list,drop_out=0.05):
        super(Inception_cat, self).__init__()

        if C_out % len(kernel_list) != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by len(kernel_list)")
        
        sub_Cout = int(C_out / len(kernel_list))

        self.conv_list = nn.ModuleList()
        for i in range(len(kernel_list)):
            self.conv_list.append(BasicNormConv(C_in = C_in, 
                                                C_out = sub_Cout, 
                                                kernel_size = kernel_list[i], 
                                                dilation = dilated_list[i],
                                                groups=1,
                                                dropout_rate = drop_out))
    def forward(self, x):
        outputs = []
        for conv in self.conv_list:
            outputs.append(conv(x))

        # 拼接所有分支的输出
        return torch.cat(outputs, 1)  # 在通道维度上拼接


class inception_sum(nn.Module):
    def __init__(self, C_in, C_out,kernel_list,dilated_list,drop_out=0.05, gelu = True, norm = True):
        super(inception_sum, self).__init__()


        self.conv_list = nn.ModuleList()
        for i in range(len(kernel_list)):
            self.conv_list.append(BasicNormConv(C_in = C_in,
                                                C_out = C_out,
                                                kernel_size = kernel_list[i],
                                                dilation = dilated_list[i],
                                                dropout_rate = drop_out,
                                                gelu=gelu,
                                                norm=norm))
    def forward(self, x):
        output = None
        for conv in self.conv_list:
            if output is None:
                output = conv(x)
            else:
                output += conv(x)

        return output  # 在通道维度上拼接


class Inception_ghost_cat(nn.Module):
    def __init__(self, C_in, C_out, kernel_list, dilated_list, drop_out=0.05):
        super(Inception_ghost_cat, self).__init__()

        if C_out % len(kernel_list) != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by len(kernel_list)")

        sub_Cout = int(C_out / len(kernel_list))

        self.conv_list = nn.ModuleList()
        for i in range(len(kernel_list)):
            self.conv_list.append(GhostConv2D(inp=C_in,
                                              oup=sub_Cout,
                                              dw_kernel=kernel_list[i],
                                              dw_dilated=dilated_list[i],
                                              drop_out = drop_out))

    def forward(self, x):
        outputs = []
        for conv in self.conv_list:
            outputs.append(conv(x))

        # 拼接所有分支的输出
        return torch.cat(outputs, 1)  # 在通道维度上拼接


class inception_ghost_sum(nn.Module):
    def __init__(self, C_in, C_out,kernel_list,dilated_list,drop_out=0.05):
        super(inception_ghost_sum, self).__init__()


        self.conv_list = nn.ModuleList()
        for i in range(len(kernel_list)):

            self.conv_list.append(GhostConv2D(inp = C_in,
                                              oup = C_out,
                                              dw_kernel=kernel_list[i],
                                              dw_dilated=dilated_list[i],
                                              drop_out = drop_out))

    def forward(self, x):
        output = None
        for conv in self.conv_list:
            if output is None:
                output = conv(x)
            else:
                output += conv(x)


        return output  # 在通道维度上拼接


class Inception_dwconv_cat(nn.Module):
    def __init__(self, C_in, C_out, kernel_list, dilated_list, drop_out=0.05):
        super(Inception_dwconv_cat, self).__init__()

        if C_out % len(kernel_list) != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by len(kernel_list)")

        sub_Cout = int(C_out / len(kernel_list))

        self.conv_list = nn.ModuleList()
        for i in range(len(kernel_list)):
            self.conv_list.append(dw_decompos_conv(inp=C_in,
                                              oup=sub_Cout,
                                              dw_kernel=kernel_list[i],
                                              dw_dilated=dilated_list[i],
                                              drop_out = drop_out))

    def forward(self, x):
        outputs = []
        for conv in self.conv_list:
            outputs.append(conv(x))

        # 拼接所有分支的输出
        return torch.cat(outputs, 1)  # 在通道维度上拼接


def Inception_dwconv_cat_test():
    # 示例参数
    batch_size = 10  # 批次大小
    channels = 4  # 通道数
    img_H = 128  # 序列长度
    img_W = 128  # 第三维度长度

    kernel_list = [3, 5, 7, 9, 11]
    dilation_list = [1, 2, 4, 8, 4]
    C_out = 40
    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据

    inception_module = Inception_dwconv_cat(C_in=channels, C_out=C_out, kernel_list=kernel_list,
                                           dilated_list=dilation_list)
    output_tensor = inception_module(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_tensor.shape}")



class inception_dwconv_sum(nn.Module):
    def __init__(self, C_in, C_out,kernel_list,dilated_list,drop_out=0.05):
        super(inception_dwconv_sum, self).__init__()


        self.conv_list = nn.ModuleList()
        for i in range(len(kernel_list)):

            self.conv_list.append(dw_decompos_conv(inp = C_in,
                                              oup = C_out,
                                              dw_kernel=kernel_list[i],
                                              dw_dilated=dilated_list[i],
                                              drop_out = drop_out))

    def forward(self, x):
        output = None
        for conv in self.conv_list:
            if output is None:
                output = conv(x)
            else:
                output += conv(x)


        return output  # 在通道维度上拼接


def inception_dwconv_sum_test():
    # 示例参数
    batch_size = 10  # 批次大小
    channels = 4  # 通道数
    img_H = 128  # 序列长度
    img_W = 128  # 第三维度长度

    kernel_list = [3, 5, 7, 9, 11]
    dilation_list = [1, 2, 4, 8, 4]
    C_out = 40
    x = torch.randn(batch_size, channels, img_H, img_W)  # 随机生成输入数据

    inception_module = inception_dwconv_sum(C_in=channels, C_out=C_out, kernel_list=kernel_list,
                                           dilated_list=dilation_list)
    output_tensor = inception_module(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_tensor.shape}")



class res_incep(nn.Module):
    def __init__(self, C_in, C_out, kernel_list, dilated_list, inception_module):
        super(res_incep,self).__init__()

        if C_out % len(kernel_list) != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by {len(kernel_list)}.")

        self.incep_path = inception_module(
            C_in=C_in,
            C_out=C_out,
            kernel_list=kernel_list,
            dilated_list=dilated_list
        )
        self.short_path = BasicNormConv(C_in, C_out, kernel_size=1)

    def forward(self, x):
        return channel_shuffle(self.incep_path(x) + self.short_path(x))


if __name__ == '__main__':
    Inception_dwconv_cat_test()
    inception_dwconv_sum_test()




