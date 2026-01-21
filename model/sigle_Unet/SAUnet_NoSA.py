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

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super(AttnBlock,self).__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0) # 卷积
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

class SAUInputConv(nn.Module):
    """
    输入卷积层：处理输入特征图，转换为指定通道数
    """
    def __init__(self, in_ch, out_ch, kernel_sizes=[3, 5]):
        super().__init__()
        self.conv = nn.Sequential(
            inception_ghost_sum(in_ch, out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1]),
            inception_ghost_sum(out_ch, out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1])
        )

    def forward(self, x):
        return self.conv(x)




class SAUDownBlock(nn.Module):
    """
    通用块：集成 InceptionGhost、Attention 和 采样层
    """

    def __init__(self, in_ch, out_ch, use_attn=False, kernel_sizes=[3, 5]):
        super().__init__()
        layers = []
        layers.append(multiScaleConvDown(in_ch, kernel_sizes))
        layers.append(inception_ghost_sum(in_ch, out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1]))
        layers.append(inception_ghost_sum(out_ch, out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1]))
        if use_attn:
            layers.append(AttnBlock(out_ch))

        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class SAUUpBlock(nn.Module):
    """
    通用块：集成 InceptionGhost、Attention 和 采样层
    """
    def __init__(self, in_ch, out_ch, use_attn=False, kernel_sizes=[3, 5]):
        super().__init__()
        layers = []
        layers.append(inception_ghost_sum(in_ch, out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1]))
        layers.append(inception_ghost_sum(out_ch, out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1]))
        if use_attn:
            layers.append(AttnBlock(out_ch))
        layers.append(multiScaleUpSample(out_ch, kernel_sizes, factor=0.5))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class SAUnetOut(nn.Module):
    def __init__(self, C_in, C_out):
        super(SAUnetOut, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SAUnetNoSA(nn.Module):
    # 修改上卷积方法
    def __init__(self, input_shape, output_shape, C_down_list):
        super(SAUnetNoSA, self).__init__()
        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, _, _ = output_shape
        kernel_sizes = [3, 5]
        atten_layer_count = len(C_down_list) - 3
        # --- 编码器 (Encoder) ---
        self.encoders = nn.ModuleList()
        self.encoders.append(SAUInputConv(self.input_channel, C_down_list[0], kernel_sizes))
        in_ch = C_down_list[0]
        for i, out_ch in enumerate(C_down_list):
            self.encoders.append(SAUDownBlock(
                in_ch, out_ch,
                use_attn=False,
                kernel_sizes=kernel_sizes
            ))
            in_ch = out_ch

        # 中心卷积层
        self.conv_center = fractal_conv(C_in=C_down_list[-1],C_out=C_down_list[-1],kernel_list = kernel_sizes,dilated_list = [1,1],inception_module = inception_sum)

        # 创建上采样路径（解码器）
        self.decoders = nn.ModuleList()
        # 注意力层计数器
        for i in range(len(C_down_list) - 1, -1, -1):
            curr_ch = C_down_list[i]
            dec_in_ch = curr_ch * 2
            self.decoders.append(SAUUpBlock(
                dec_in_ch, curr_ch,
                use_attn=False,
                kernel_sizes=kernel_sizes
            ))

        self.tail = SAUnetOut(C_down_list[0]//2 + C_down_list[0], self.output_channel)

    def forward(self, x):
        # 编码器路径
        encoder_outs = []
        for layer in self.encoders:
            x = layer(x)
            encoder_outs.append(x)

        # 中心处理
        down4_out = encoder_outs[-1]
        center_out = self.conv_center(down4_out)
        x = torch.cat([center_out, down4_out], dim=1)

        for i in range(len(self.decoders)):
            # 上采样卷积
            x = self.decoders[i](x)
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
    batch_size = 8
    img_H = 256
    img_W = 256
    input_data = torch.randn(batch_size, 6, img_H, img_W).to(device)

    BTM_ghost_UNet_input_shape = [input_data.shape[1], input_data.shape[2], input_data.shape[3]]
    BTM_ghost_UNet_output_shape = [1, input_data.shape[2], input_data.shape[3]]
    C_down_list = [64, 128, 256, 512]
    model = SAUnet(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list).to(device)
    x = input_data
    get_gpu_info.print_gpu_memory("Conv3x3_DownSample GPU info", x, model)



if __name__ == '__main__':
    SAUnet_test()
    # BTM_ghost_UNet_test()





