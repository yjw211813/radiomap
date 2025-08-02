import torch
import sys
import os
##  在多尺度卷积基础上，将多尺度卷积引入到分形网络中
# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.sub_block.conv2D_block import *

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
# 变换到相同形状进行加和形式


class BTM_Net_v4(nn.Module):
    def __init__(self,input_shape, output_shape,C_list):
        super(BTM_Net_v4, self).__init__()
        self.input_channel , self.input_H , self.input_W = input_shape
        self.output_channel , self.output_H , self.output_W = output_shape

        C_list = torch.cat((C_list, torch.tensor([self.output_channel], dtype=torch.int32)))
        # 构建编码器
        layers = []
        layers.append(multi_scale_block2D(C_in=self.input_channel, C_out=C_list[0], kernel_sizes=[1, 3, 5, 7], dilated_num=1))
        for i in range(len(C_list) - 1):
            layers.append(multi_scale_block2D(C_in=C_list[i], C_out=C_list[i + 1], kernel_sizes=[1, 3, 5, 7], dilated_num=1))
        self.encoder = nn.Sequential(*layers)
        self.gelu = nn.GELU()

    def forward(self, condition_map):

        resized_map = F.interpolate(condition_map, size=(self.output_H, self.output_W), mode='bilinear', align_corners=True)
        out_map = self.encoder(resized_map)

        return out_map


class BTM_multi_scale_v6(nn.Module):
    # 修改上卷积方法
    def __init__(self, input_shape, output_shape, C_down_list,attn_params):
        super(BTM_multi_scale_v6, self).__init__()
        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, _, _ = output_shape
        kernel_sizes = [3, 5, 7, 9]
        # 创建下采样路径（编码器）
        self.encoder = nn.ModuleList()
        in_ch = self.input_channel
        for out_ch in C_down_list:
            self.encoder.append(nn.Sequential(
                multi_scale_block2D(C_in=in_ch, C_out=out_ch, kernel_sizes=kernel_sizes, dilated_num=1),
                Conv_DownSampling2D(out_ch)
            ))
            in_ch = out_ch

        # 中心卷积层
        self.conv_center = Fractal_multi_scale2D(input_channel=C_down_list[-1],output_channel=C_down_list[-1])

        # 创建上采样路径（解码器）
        self.decodes = nn.ModuleList()
        for i in range(len(C_down_list), 0, -1):
            i = i -1
            self.decodes.append(
                nn.Sequential(
                    multi_scale_block2D(C_in=C_down_list[i] + C_down_list[i],C_out=C_down_list[i],kernel_sizes=kernel_sizes,dilated_num=1),
                    ConvTranspose_UpSam(C_down_list[i])
                )
            )
        # 创建注意力模块
        self.attentions = nn.ModuleList()
        attn_channels = [
            C_down_list[2] // 4,  # 对应第4层
            C_down_list[1] // 4,  # 对应第3层
            C_down_list[0] // 4,  # 对应第2层
            C_down_list[0] // 4  # 对应第1层
        ]
        attn_factors = [8, 4, 2, 1]  # 空间尺寸缩小因子
        self.repeat_factors = [4,4,4,2]
        for ch, factor, param in zip(attn_channels, attn_factors, attn_params):
            self.attentions.append(BTM_Net_v4(
                input_shape,
                [ch, self.input_H // factor, self.input_W // factor],
                param
            ))

        # 输出层
        now_ch = C_down_list[0] // 2
        self.tail = nn.Sequential(
            nn.GroupNorm(now_ch // 4, now_ch),
            Swish(),
            nn.Conv2d(now_ch, self.output_channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        # 解码器路径
        attn_outputs = []
        for i in range(len(self.attentions)):
            attn_outputs.append(self.attentions[i](x).repeat(1, self.repeat_factors[i], 1, 1))

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
            # 应用注意力机制
            x = x * attn_outputs[i]
            # 跳跃连接（拼接编码器特征）
            skip_idx = len(encoder_outs) - 2 - i
            if skip_idx >= 0:
                x = torch.cat([x, encoder_outs[skip_idx]], dim=1)

        # 最终输出层
        return self.tail(x * attn_outputs[-1])

    def load_weights(self, checkpoint_path):
        """加载预训练权重"""
        self.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded weights from {checkpoint_path}")


def BTM_multi_scale_v6_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 8
    feature_num = 4
    img_H = 256
    img_W = 256
    input_data = torch.randn(batch_size, 4, img_H, img_W).to(device)

    BTM_ghost_UNet_input_shape = [input_data.shape[1], input_data.shape[2], input_data.shape[3]]
    BTM_ghost_UNet_output_shape = [1, input_data.shape[2], input_data.shape[3]]
    C_down_list = [64, 128, 256, 512]
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    net = BTM_multi_scale_v6(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)
    output = net(input_data)
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    BTM_multi_scale_v6_test()
    # BTM_ghost_UNet_test()





