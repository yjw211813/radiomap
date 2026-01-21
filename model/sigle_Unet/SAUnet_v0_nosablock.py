import torch
import torch.nn as nn
import sys
import os
##  在多尺度卷积基础上，将多尺度卷积引入到分形网络中
# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.sub_block.top_conv import SK_Channel_atten2D,fractal_conv
from model.sub_block.mid_conv import inception_ghost_sum,inception_sum
from model.sub_block.low_conv import multiScaleConvDown,multiScaleUpSample
from torch.nn import functional as F
from model.sub_block.statistic_tools import gpu_statistic



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
# 变换到相同形状进行加和形式


class SAUnet_old_nosa(nn.Module):
    # 修改上卷积方法
    def __init__(self, input_shape, output_shape, C_down_list):
        super(SAUnet_old_nosa, self).__init__()
        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, _, _ = output_shape
        kernel_sizes = [3, 5, 7, 9]
        # 创建下采样路径（编码器）
        self.encoder = nn.ModuleList()
        in_ch = self.input_channel
        for out_ch in C_down_list:
            self.encoder.append(nn.Sequential(
                inception_ghost_sum(C_in=in_ch, C_out=out_ch,  kernel_list=kernel_sizes, dilated_list=[1,1,1,1]),
                multiScaleConvDown(out_ch,kernel_sizes)
            ))
            in_ch = out_ch

        # 中心卷积层
        self.conv_center = fractal_conv(C_in=C_down_list[-1],C_out=C_down_list[-1],kernel_list = kernel_sizes,dilated_list = [1,1,1,1],inception_module = inception_sum)

        # 创建上采样路径（解码器）
        self.decodes = nn.ModuleList()
        for i in range(len(C_down_list), 0, -1):
            i = i -1
            self.decodes.append(
                nn.Sequential(
                    inception_ghost_sum(C_in=C_down_list[i] + C_down_list[i],C_out=C_down_list[i], kernel_list=kernel_sizes, dilated_list=[1,1,1,1]),
                    multiScaleUpSample(C_down_list[i],kernel_sizes,factor=0.5)
                )
            )

        # 输出层
        now_ch = C_down_list[0] // 2
        self.tail = nn.Sequential(
            nn.GroupNorm(now_ch // 4, now_ch),
            Swish(),
            nn.Conv2d(now_ch, self.output_channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        # 解码器路径

        # 编码器路径
        encoder_outs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outs.append(x)

        # 中心处理
        down4_out = encoder_outs[-1]
        center_out = self.conv_center(down4_out)
        x = torch.cat([center_out, down4_out], dim=1)

        for i in range(len(self.decodes)):
            # 上采样卷积
            x = self.decodes[i](x)
            # 跳跃连接（拼接编码器特征）
            skip_idx = len(encoder_outs) - 2 - i
            if skip_idx >= 0:
                x = torch.cat([x, encoder_outs[skip_idx]], dim=1)

        # 最终输出层
        return self.tail(x)

    def load_weights(self, checkpoint_path):
        """加载预训练权重"""
        self.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded weights from {checkpoint_path}")


def SAUnet_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    batch_size = 16
    img_H = 256
    img_W = 256
    input_data = torch.randn(batch_size, 6, img_H, img_W).to(device)

    BTM_ghost_UNet_input_shape = [input_data.shape[1], input_data.shape[2], input_data.shape[3]]
    BTM_ghost_UNet_output_shape = [1, input_data.shape[2], input_data.shape[3]]
    C_down_list = [32, 64, 128, 256]
    model = SAUnet_old_nosa(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list).to(device)
    x = input_data
    get_gpu_info.print_gpu_memory("Conv3x3_DownSample GPU info", x, model)



if __name__ == '__main__':
    SAUnet_test()
    # BTM_ghost_UNet_test()

