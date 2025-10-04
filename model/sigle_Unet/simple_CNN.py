import torch
import torch.nn as nn
import sys
import os
##  在多尺度卷积基础上，将多尺度卷积引入到分形网络中
# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.flow_matching_model.volecity_predict import LSKNet,AttnBlock
from model.sub_block.top_conv import SK_Channel_atten2D,fractal_conv
from model.sub_block.mid_conv import inception_ghost_sum,inception_sum
from model.sub_block.low_conv import multiScaleConvDown,multiScaleUpSample,Conv_DownSampling2D,BasicNormConv
from torch.nn import functional as F
from model.sub_block.statistic_tools import gpu_statistic

#  处理 下采样 处理 下采样 处理 下采样
#  存
class resConv(nn.Module):
    def __init__(self,C_in,C_out,kernel_list = [3,5],dilated_list = [1,1], attn="Attn", LSK_kernels = [5,7], LSK_dilats = [1,1], LSK_mid_kernel = 7):
        super(resConv, self).__init__()
        self.conv_begin = BasicNormConv(C_in = C_in,
                                                C_out = C_out,
                                                kernel_size = kernel_list[0],
                                                dilation = dilated_list[0],
                                                gelu=True,
                                                norm=True)
        self.conv_end = BasicNormConv(C_in = C_out,
                                                C_out = C_out,
                                                kernel_size = kernel_list[1],
                                                dilation = dilated_list[1],
                                                gelu=True,
                                                norm=True)
        if C_in != C_out:
            self.shortcut = BasicNormConv(C_in = C_in, C_out = C_out, kernel_size = 1,gelu=False,norm = False)
        else:
            self.shortcut = nn.Identity()

        if attn == "LSKNet":
            self.attn = LSKNet(C_out, kernel_mid = LSK_mid_kernel, kernel_list = LSK_kernels, dilated_list = LSK_dilats)
        elif attn == "Attn":
            self.attn = AttnBlock(C_out)
        else:
            self.attn = nn.Identity()


    def forward(self, x):
        out =self.conv_end( self.conv_begin(x))
        out = self.attn(out + self.shortcut(x))
        return out

class SANet(nn.Module):
    def __init__(self,C_in, output_shape,kernel_list = [3,5], dilated_list = [1,1],attn = "LSKNet"):
        super(SANet, self).__init__()
        self.input_channel  = C_in

        self.output_channel , self.output_H , self.output_W = output_shape

        self.encoder = resConv(C_in=self.input_channel,C_out=self.output_channel,kernel_list=kernel_list,dilated_list=dilated_list,attn=attn)


    def forward(self, inps,condition):

        resized_map = F.interpolate(inps, size=(self.output_H, self.output_W), mode='bilinear', align_corners=True)
        out_map = self.encoder(torch.cat([resized_map, condition], dim=1))

        return out_map

def SANet_test():
    """测试resConv模块"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    # 测试参数
    batch_size = 16
    img_H = 256
    img_W = 256
    C_in = 6
    C_in2 = 34
    C_out = 32
    # 创建输入数据
    input_data1 = torch.randn(batch_size, C_in, img_H, img_W).to(device)
    input_data2 = torch.randn(batch_size, C_in2, 32, 32).to(device)
    input_shape = [C_in, img_H, img_W]
    output_shape = [C_out, 32, 32]

    # 测试用例1: 基础配置 (使用Attn)
    print("\n1. Testing resConv with Attn (C_in != C_out):")
    model = SANet(C_in = C_in+C_in2,output_shape= output_shape).to(device)

    get_gpu_info.print_gpu_memory(description = "Conv3x3_DownSample GPU info", x = input_data1,model = model,temb=input_data2)




class SAUnetDown(nn.Module):
    def __init__(self, C_in, C_out,kernel_list = [3,5], dilated_list = [1,1],attn = "LSKNet"):
        super(SAUnetDown, self).__init__()
        max_flag = False
        kernel_sizes = [3,5,7]
        self.conv = nn.Sequential(
            nn.MaxPool2d(2)  if max_flag else multiScaleConvDown(C_in,kernel_sizes),
            resConv(C_in=C_in,C_out=C_out,kernel_list=kernel_list,dilated_list=dilated_list,attn=attn)
        )

    def forward(self,x):
        return self.conv (x)

class SAUnetUp(nn.Module):
    def __init__(self, C_in, C_out,outSize,kernel_list = [3,5], dilated_list = [1,1],attn = "Attn"):
        super(SAUnetUp, self).__init__()
        simple_flag = False

        kernel_sizes = [3,5,7]
        outShape = [C_in // 2,outSize,outSize]
        self.up =  nn.ConvTranspose2d(C_in, C_in // 2, kernel_size=2, stride=2)  if simple_flag else multiScaleUpSample(C_in,kernel_sizes,factor=0.5)
        self.SA_atten = SANet(64 + C_in // 2,outShape,attn = "Identity")
        self.conv = resConv(C_in=C_in,C_out=C_out,kernel_list=kernel_list,dilated_list=dilated_list,attn=attn)


    def forward(self, xDecode, xEncode,inps):
        xDecode = self.up(xDecode) * self.SA_atten(inps,xEncode)

        # input is CHW
        diffY = xEncode.size()[2] - xDecode.size()[2]
        diffX = xEncode.size()[3] - xDecode.size()[3]

        xDecode = F.pad(xDecode, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        out = self.conv (torch.cat([xEncode, xDecode], dim=1))

        return out

class SAUnetOut(nn.Module):
    def __init__(self, C_in, C_out):
        super(SAUnetOut, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class SAUnetForProcess(nn.Module):
    def __init__(self,input_shape, output_shape):
        super(SAUnetForProcess, self).__init__()
        self.input_channel, self.input_H, self.input_W = input_shape # (batchsize,6,256,256)
        self.output_channel, _, _ = output_shape


        self.inc = resConv(self.input_channel, 64,attn = "LSKNet")      # 256
        self.down1 = SAUnetDown(64, 128,attn = "LSKNet")           # 128
        self.down2 = SAUnetDown(128, 256,attn = "Attn")          # 64
        self.down3 = SAUnetDown(256, 512,attn = "Attn")          # 32
        self.down4 = SAUnetDown(512, 1024,attn = "Attn")         # 16

        self.up1 = SAUnetUp(1024, 512,32,attn = "Attn")             # 32
        self.up2 = SAUnetUp(512, 256,64,attn = "Attn")              # 64
        self.up3 = SAUnetUp(256, 128,128,attn = "LSKNet")            # 128
        self.up4 = SAUnetUp(128, 64,256,attn = "LSKNet")             # 256
        self.outc = SAUnetOut(64, self.output_channel)

    def forward(self,inp):
        x = inp
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4,x1)
        x = self.up2(x, x3,x1)
        x = self.up3(x, x2,x1)
        x = self.up4(x, x1,x1)
        x = self.outc(x)
        return x

def SAUnetForProcess_test():
    """测试resConv模块"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    # 测试参数
    batch_size = 16
    img_H = 256
    img_W = 256
    C_in = 6
    C_out = 1
    # 创建输入数据
    input_data1 = torch.randn(batch_size, C_in, img_H, img_W).to(device)
    input_shape = [C_in, img_H, img_W]
    output_shape = [C_out, img_H, img_W]

    # 测试用例1: 基础配置 (使用Attn)
    print("\n1. nihaoTesting resConv with Attn (C_in != C_out):")
    model = SAUnetForProcess(input_shape = input_shape,output_shape= output_shape).to(device)

    get_gpu_info.print_gpu_memory(description = "Conv3x3_DownSample GPU info", x = input_data1,model = model)



def resConv_test():
    """测试resConv模块"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    # 测试参数
    batch_size = 16
    img_H = 64
    img_W = 64
    C_in = 256
    C_out = 512
    kernel_list = [3, 5]
    dilated_list = [1, 1]

    # 创建输入数据
    input_data = torch.randn(batch_size, C_in, img_H, img_W).to(device)

    # 测试用例1: 基础配置 (使用Attn)
    print("\n1. Testing resConv with Attn (C_in != C_out):")
    model = resConv(C_in=C_in,
                     C_out=C_out,
                     kernel_list=kernel_list,
                     dilated_list=dilated_list,
                     attn="Attn").to(device)

    get_gpu_info.print_gpu_memory("Conv3x3_DownSample GPU info", input_data, model)

def SAUnetUp_test():
    """测试resConv模块"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    # 测试参数
    batch_size = 16
    img_H = 64
    img_W = 64
    C_in = 512
    C_out = 256
    # 创建输入数据
    input_data1 = torch.randn(batch_size, C_in, img_H, img_W).to(device)
    input_data2 = torch.randn(batch_size, C_out, 128, 128).to(device)
    input_data3 = torch.randn(batch_size, 6, 256, 256).to(device)
    # 测试用例1: 基础配置 (使用Attn)
    print("\n1. Testing resConv with Attn (C_in != C_out):")
    model = SAUnetUp(C_in=C_in,
                     C_out=C_out,
                     outSize=128).to(device)

    get_gpu_info.print_gpu_memory(description = "Conv3x3_DownSample GPU info", x = input_data1,model = model,temb = input_data2,condition = input_data3)



def SAUnetDown_test():
    """测试resConv模块"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    # 测试参数
    batch_size = 16
    img_H = 64
    img_W = 64
    C_in = 256
    C_out = 512

    # 创建输入数据
    input_data = torch.randn(batch_size, C_in, img_H, img_W).to(device)

    # 测试用例1: 基础配置 (使用Attn)
    print("\n1. Testing SAUnetDown_test ")
    model = SAUnetDown(C_in=C_in,
                     C_out=C_out).to(device)

    get_gpu_info.print_gpu_memory("Conv3x3_DownSample GPU info", input_data, model)

def SAUnetOut_test():
    """测试resConv模块"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    # 测试参数
    batch_size = 16
    img_H = 64
    img_W = 64
    C_in = 64
    C_out = 1
    # 创建输入数据
    input_data1 = torch.randn(batch_size, C_in, img_H, img_W).to(device)

    # 测试用例1: 基础配置 (使用Attn)
    print("\n1. Testing resConv with Attn (C_in != C_out):")
    model = SAUnetOut(C_in=C_in,
                     C_out=C_out).to(device)

    get_gpu_info.print_gpu_memory(description = "Conv3x3_DownSample GPU info", x = input_data1,model = model)



if __name__ == '__main__':
    SAUnetForProcess_test()