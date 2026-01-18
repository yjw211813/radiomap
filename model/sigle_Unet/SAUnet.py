import torch
import torch.nn as nn
import sys
import os
##  在多尺度卷积基础上，将多尺度卷积引入到分形网络中
# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.sub_block.top_conv import SK_Channel_atten2D,fractal_conv
from model.sub_block.mid_conv import inception_ghost_sum,inception_sum
from model.sub_block.low_conv import multiScaleConvDown,multiScaleUpSample,BasicGhostConv
from torch.nn import functional as F
from model.sub_block.statistic_tools import gpu_statistic



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
# 变换到相同形状进行加和形式


class ImprovedBTM_Net(nn.Module):
    def __init__(self, input_shape, output_shape, C_list, use_attention=True, use_residual=True):
        super(ImprovedBTM_Net, self).__init__()
        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, self.output_H, self.output_W = output_shape
        self.use_attention = use_attention
        self.use_residual = use_residual

        C_list = torch.cat((C_list, torch.tensor([self.output_channel], dtype=torch.int32)))

        # 构建编码器 - 改进的层次结构
        self.encoder_layers = nn.ModuleList()

        # 输入层
        self.encoder_layers.append(
            ImprovedInceptionGhostBlock(
                C_in=self.input_channel,
                C_out=C_list[0],
                kernel_list=[3, 5, 7],
                dilated_list=[1, 2, 3],  # 增加空洞卷积增强感受野
                use_attention=use_attention,
                use_residual=use_residual
            )
        )

        # 中间层
        for i in range(len(C_list) - 1):
            self.encoder_layers.append(
                ImprovedInceptionGhostBlock(
                    C_in=C_list[i],
                    C_out=C_list[i + 1],
                    kernel_list=[3, 5, 7],
                    dilated_list=[1, 2, 3],
                    use_attention=use_attention,
                    use_residual=use_residual
                )
            )

        # 输出处理模块
        self.output_processor = OutputProcessor(
            C_in=C_list[-1],
            C_out=self.output_channel,
            use_attention=use_attention
        )

        # 门控机制
        if use_attention:
            self.gate_controller = GateController(
                input_channel=self.input_channel,
                output_channel=self.output_channel,
                spatial_size=(self.output_H, self.output_W)
            )

    def forward(self, condition_map):
        # 下采样输入到目标尺寸
        resized_map = F.interpolate(
            condition_map,
            size=(self.output_H, self.output_W),
            mode='bilinear',
            align_corners=True
        )

        # 编码过程
        x = resized_map
        for layer in self.encoder_layers:
            x = layer(x)

        # 门控控制（如果使用）
        if self.use_attention and hasattr(self, 'gate_controller'):
            gate_weights = self.gate_controller(condition_map, resized_map)
            x = x * gate_weights

        # 输出处理
        out_map = self.output_processor(x)

        return out_map


class ImprovedInceptionGhostBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_list, dilated_list, use_attention=True, use_residual=True, drop_out=0.05):
        super(ImprovedInceptionGhostBlock, self).__init__()
        self.use_residual = use_residual
        self.use_attention = use_attention

        # 多尺度卷积分支
        self.conv_branches = nn.ModuleList()
        for i in range(len(kernel_list)):
            self.conv_branches.append(
                EnhancedGhostConv(
                    inp=C_in,
                    oup=C_out // len(kernel_list),  # 平均分配通道
                    dw_kernel=kernel_list[i],
                    dw_dilated=dilated_list[i],
                    drop_out=drop_out
                )
            )

        # 残差连接
        if use_residual and C_in == C_out:
            self.residual = nn.Identity()
        elif use_residual:
            self.residual = nn.Sequential(
                nn.Conv2d(C_in, C_out, 1, bias=False),
                nn.BatchNorm2d(C_out)
            )
        else:
            self.residual = None

        # 注意力机制
        if use_attention:
            self.channel_attention = ChannelAttention(C_out)
            self.spatial_attention = SpatialAttention()

        # 归一化和激活
        self.norm = nn.BatchNorm2d(C_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop_out)

    def forward(self, x):
        residual = x

        # 多分支处理
        outputs = []
        for branch in self.conv_branches:
            outputs.append(branch(x))

        # 合并分支（使用拼接而不是相加，保留更多信息）
        if len(outputs) > 1:
            x = torch.cat(outputs, dim=1)
        else:
            x = outputs[0]

        # 残差连接
        if self.use_residual and self.residual is not None:
            if residual.shape[1] != x.shape[1]:
                residual = self.residual(residual)
            x = x + residual

        # 注意力机制
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)

        # 后处理
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x


class EnhancedGhostConv(nn.Module):
    def __init__(self, inp, oup, dw_kernel, dw_dilated, channel_kernel_size=1, ratio=2, drop_out=0.05):
        super(EnhancedGhostConv, self).__init__()
        self.ghost_conv = BasicGhostConv(
            inp=inp, oup=oup,
            dw_kernel=dw_kernel, dw_dilated=dw_dilated,
            channel_kernel_size=channel_kernel_size, ratio=ratio
        )
        self.norm = nn.BatchNorm2d(oup)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop_out)

    def forward(self, x):
        x = self.ghost_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class OutputProcessor(nn.Module):
    """专门处理输出的模块，确保输出在[0,1]范围内"""

    def __init__(self, C_in, C_out, use_attention=True):
        super(OutputProcessor, self).__init__()

        # 最终卷积层
        self.final_conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.GELU()
        )

        # 输出激活 - 使用Sigmoid确保[0,1]范围
        self.output_activation = nn.Sigmoid()

        # 可选的最终注意力
        if use_attention:
            self.final_attention = SpatialAttention()
        else:
            self.final_attention = None

    def forward(self, x):
        x = self.final_conv(x)

        if self.final_attention is not None:
            x = self.final_attention(x)

        # 应用Sigmoid确保输出在[0,1]范围内
        x = self.output_activation(x)

        return x


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class GateController(nn.Module):
    """门控控制器，根据输入条件调整输出"""

    def __init__(self, input_channel, output_channel, spatial_size):
        super(GateController, self).__init__()
        self.spatial_H, self.spatial_W = spatial_size

        # 处理原始输入特征
        self.input_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.spatial_H, self.spatial_W)),
            nn.Conv2d(input_channel, output_channel, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_channel, output_channel, 1),
            nn.Sigmoid()  # 输出在[0,1]范围内作为权重
        )

    def forward(self, original_input, resized_input):
        gate_weights = self.input_processor(original_input)
        return gate_weights






class SAUnet(nn.Module):
    # 修改上卷积方法
    def __init__(self, input_shape, output_shape, C_down_list,attn_params):
        super(SAUnet, self).__init__()
        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, _, _ = output_shape
        kernel_sizes = [3, 5]
        # 创建下采样路径（编码器）
        self.encoder = nn.ModuleList()
        in_ch = self.input_channel
        for out_ch in C_down_list:
            self.encoder.append(nn.Sequential(
                inception_ghost_sum(C_in=in_ch, C_out=out_ch,  kernel_list=kernel_sizes, dilated_list=[1,1]),
                inception_ghost_sum(C_in=out_ch, C_out=out_ch, kernel_list=kernel_sizes, dilated_list=[1, 1]),
                multiScaleConvDown(out_ch,kernel_sizes)
            ))
            in_ch = out_ch

        # 中心卷积层
        self.conv_center = fractal_conv(C_in=C_down_list[-1],C_out=C_down_list[-1],kernel_list = kernel_sizes,dilated_list = [1,1],inception_module = inception_sum)

        # 创建上采样路径（解码器）
        self.decodes = nn.ModuleList()
        for i in range(len(C_down_list), 0, -1):
            i = i -1
            self.decodes.append(
                nn.Sequential(
                    inception_ghost_sum(C_in=C_down_list[i] + C_down_list[i],C_out=C_down_list[i] + C_down_list[i], kernel_list=kernel_sizes, dilated_list=[1,1]),
                    inception_ghost_sum(C_in=C_down_list[i] + C_down_list[i], C_out=C_down_list[i],kernel_list=kernel_sizes, dilated_list=[1, 1]),
                    multiScaleUpSample(C_down_list[i],kernel_sizes,factor=0.5)
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
            self.attentions.append(ImprovedBTM_Net(
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
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    model = SAUnet(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)
    x = input_data
    get_gpu_info.print_gpu_memory("Conv3x3_DownSample GPU info", x, model)

def ImprovedBTM_Net_test():
    # 初始化网络
    net = ImprovedBTM_Net(
        input_shape=(3, 256, 256),
        output_shape=(1, 128, 128),
        C_list=torch.tensor([32, 64, 128]),
        use_attention=True,
        use_residual=True
    )

    # 前向传播
    input_tensor = torch.randn(2, 3, 256, 256)
    output = net(input_tensor)
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")  # 应该在[0,1]范围内


if __name__ == '__main__':
    # SAUnet_test()
    ImprovedBTM_Net_test()
    # BTM_ghost_UNet_test()





