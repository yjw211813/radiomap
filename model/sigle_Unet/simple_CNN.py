import torch
import torch.nn as nn
import sys
import os
##  在多尺度卷积基础上，将多尺度卷积引入到分形网络中
# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.sub_block.top_conv import SK_Channel_atten2D,fractal_conv
from model.sub_block.mid_conv import inception_ghost_sum,inception_sum
from model.sub_block.low_conv import multiScaleConvDown,multiScaleUpSample,Conv_DownSampling2D,BasicNormConv
from torch.nn import functional as F


class simple_CNN(nn.Module):
    def __init__(self,input_shape, output_shape,C_down_list):
        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, _, _ = output_shape

        self.encoder = nn.ModuleList()
        in_ch = self.input_channel
        for out_ch in C_down_list:
            self.encoder.append(nn.Sequential(
                BasicNormConv(C_in=in_ch, C_out=out_ch,  kernel_size=3, dilation=1,dropout_rate = 0,groups = 1,gelu=True,norm = True),
                Conv_DownSampling2D(out_ch)
            ))
            in_ch = out_ch
        # 中心卷积层
        self.conv_center = BasicNormConv(C_in=C_down_list[-1], C_out=C_down_list[-1],  kernel_size=3, dilation=1,dropout_rate = 0,groups = 1,gelu=True,norm = True)

        # 创建上采样路径（解码器）
        self.decodes = nn.ModuleList()
        for i in range(len(C_down_list), 0, -1):
            i = i -1
            self.decodes.append(
                nn.Sequential(
                    inception_ghost_sum(C_in=C_down_list[i] + C_down_list[i],C_out=C_down_list[i], kernel_list=[1, 3, 5, 7], dilated_list=[1,1,1,1]),
                    multiScaleUpSample(C_down_list[i],kernel_sizes,factor=0.5)
                )
            )





    def forward(self):
        print(2)