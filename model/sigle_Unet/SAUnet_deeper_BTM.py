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


class BTM_Net(nn.Module):
    def __init__(self,input_shape, output_shape,C_list):
        super(BTM_Net, self).__init__()
        self.input_channel , self.input_H , self.input_W = input_shape
        self.output_channel , self.output_H , self.output_W = output_shape

        C_list = torch.cat((C_list, torch.tensor([self.output_channel], dtype=torch.int32)))
        # 构建编码器
        layers = []
        layers.append(inception_ghost_sum(C_in=self.input_channel, C_out=C_list[0], kernel_list=[3, 5, 7], dilated_list=[1,1,1]))
        for i in range(len(C_list) - 1):
            layers.append(inception_ghost_sum(C_in=C_list[i], C_out=C_list[i + 1], kernel_list=[3, 5, 7],  dilated_list=[1,1,1]))
        self.encoder = nn.Sequential(*layers)
        self.gelu = nn.GELU()

    def forward(self, condition_map):

        out_map = self.encoder(condition_map)

        return out_map


class SAUnet_deeper_BTM(nn.Module):
    def __init__(self, input_shape, output_shape, C_down_list, attn_params):
        super(SAUnet_deeper_BTM, self).__init__()
        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, _, _ = output_shape
        kernel_sizes = [3, 5]

        # --- 编码器 (Encoder) ---
        self.encoder = nn.ModuleList()
        in_ch = self.input_channel
        for out_ch in C_down_list:
            self.encoder.append(nn.Sequential(
                inception_ghost_sum(C_in=in_ch, C_out=out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1]),
                inception_ghost_sum(C_in=out_ch, C_out=out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1]),
                multiScaleConvDown(out_ch, kernel_sizes)
            ))
            in_ch = out_ch

        # --- 中心层 (Bottleneck) ---
        self.conv_center = fractal_conv(C_in=C_down_list[-1], C_out=C_down_list[-1],
                                        kernel_list=kernel_sizes, dilated_list=[1, 1],
                                        inception_module=inception_sum)

        # --- 解码器 (Decoder) ---
        self.decodes = nn.ModuleList()
        # 注意：这里假设 C_down_list 长度对应层数
        for i in range(len(C_down_list), 0, -1):
            i = i - 1
            # 解码器的输入通常是上一层的输出 + 跳跃连接的通道数
            # 这里的 C_in 设置为 2倍通道，说明是为拼接后的特征准备的
            self.decodes.append(
                nn.Sequential(
                    inception_ghost_sum(C_in=C_down_list[i] + C_down_list[i], C_out=C_down_list[i] + C_down_list[i],
                                        kernel_list=kernel_sizes, dilated_list=[1, 1]),
                    inception_ghost_sum(C_in=C_down_list[i] + C_down_list[i], C_out=C_down_list[i],
                                        kernel_list=kernel_sizes, dilated_list=[1, 1]),
                    multiScaleUpSample(C_down_list[i], kernel_sizes, factor=0.5)
                )
            )

        # --- BTM 注意力模块 ---
        self.attentions = nn.ModuleList()
        # 注意力通道配置 (从深层到浅层排列)
        attn_channels = [
            C_down_list[2] // 4,  # Level 3 (Deepest, used at center/first decode)
            C_down_list[1] // 4,  # Level 2
            C_down_list[0] // 4,  # Level 1
            C_down_list[0] // 4  # Level 0 (Shallowest)
        ]
        # 对应输入图像的下采样因子，确保生成的大小与特征图一致
        attn_factors = [8, 4, 2, 1]

        for ch, factor, param in zip(attn_channels, attn_factors, attn_params):
            self.attentions.append(BTM_Net(
                input_shape,
                [ch, self.input_H // factor, self.input_W // factor],
                param
            ))

        # --- 输出层 ---
        now_ch = C_down_list[0] // 2
        self.tail = nn.Sequential(
            nn.GroupNorm(now_ch // 4, now_ch),
            Swish(),
            nn.Conv2d(now_ch, self.output_channel, 3, stride=1, padding=1)
        )

    def forward(self, x):

        # 2. 编码器路径
        encoder_outs = []
        enc_x = x
        for layer in self.encoder:
            enc_x = layer(enc_x)
            encoder_outs.append(enc_x)

        # 3. 中心处理 (Bottleneck)
        down4_out = encoder_outs[-1]
        center_out = self.conv_center(down4_out)

        # --- 对齐并应用注意力到最深层 Skip Connection ---
        skip_feat_center = down4_out
        att_map_center = attn_outputs[0]

        # [关键修改]：使用插值确保 H 和 W 完全匹配
        if att_map_center.shape[2:] != skip_feat_center.shape[2:]:
            att_map_center = F.interpolate(
                att_map_center,
                size=skip_feat_center.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # 动态匹配通道数 (C)
        if skip_feat_center.shape[1] != att_map_center.shape[1]:
            repeat_times = skip_feat_center.shape[1] // att_map_center.shape[1]
            att_map_center = att_map_center.repeat(1, repeat_times, 1, 1)

        skip_feat_center = skip_feat_center * att_map_center
        x = torch.cat([center_out, skip_feat_center], dim=1)

        # 4. 解码器路径
        for i, decoder_layer in enumerate(self.decodes):
            x = decoder_layer(x)

            skip_idx = len(encoder_outs) - 2 - i
            if skip_idx >= 0:
                skip_feat = encoder_outs[skip_idx]

                if i + 1 < len(attn_outputs):
                    att_map = attn_outputs[i + 1]

                    # [关键修改]：同样对后续层级进行空间尺寸对齐
                    if att_map.shape[2:] != skip_feat.shape[2:]:
                        att_map = F.interpolate(
                            att_map,
                            size=skip_feat.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    # 匹配通道数
                    if skip_feat.shape[1] != att_map.shape[1]:
                        r_times = skip_feat.shape[1] // att_map.shape[1]
                        att_map = att_map.repeat(1, r_times, 1, 1)

                    skip_feat = skip_feat * att_map

                x = torch.cat([x, skip_feat], dim=1)

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

    C_down_list = [32, 64, 128, 256]
    C_list_attn = torch.tensor([32, 64])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    model = SAUnet_deeper_BTM(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)
    x = input_data
    get_gpu_info.print_gpu_memory("Conv3x3_DownSample GPU info", x, model)



if __name__ == '__main__':
    SAUnet_test()
    # BTM_ghost_UNet_test()





