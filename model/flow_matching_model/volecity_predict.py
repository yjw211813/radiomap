
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from model.sub_block.conv2D_block import Fractal_multi_scale2D,multi_scale_block2D,Conv_DownSampling2D,ConvTranspose_UpSam



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 这个暂时不动
class TimeEmbedding(nn.Module):
    # d_model 是频率嵌入维度
    def __init__(self, T, d_model, img_H, img_W):
        self.img_H, self.img_W = img_H, img_W
        assert d_model % 2 == 0
        self.d_model = d_model
        self.T = T
        super().__init__()
        # 验证维度是偶数
        assert d_model % 2 == 0, "d_model must be even"
        # 预计算频率基础参数 (固定)
        freqs = torch.pow(
            10000,
            torch.arange(0, d_model // 2) / (d_model // 2)
        )
        self.register_buffer('freqs', freqs)


        self.timembedding = nn.Sequential(
            nn.Linear(d_model, self.img_H),
            Swish(),
            nn.Linear(self.img_H, self.img_H*self.img_W),
        )

    def forward(self, t):
        """
        前向传播
        Args:
            t (torch.Tensor): 时间向量，形状为 [B]，值在[0,1]范围内
        Returns:
            torch.Tensor: 时间嵌入特征图，形状为 [B, 1, H, W]
        """
        # 1. 将时间扩展到模型范围 [0, self.T]
        scaled_t = t * self.T  # [B]
        if scaled_t.dim() == 0:
            scaled_t = scaled_t.unsqueeze(0)  # 如果是标量，转为1维
        # 2. 计算正弦/余弦分量 [B, d_model//2]
        emb = scaled_t[:, None] / self.freqs  # [B, d_model//2]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)

        # 3. 拼接特征 [B, d_model]
        emb = torch.cat([sin_emb, cos_emb], dim=-1)

        # 4. 通过神经网络调整 [B, d_model] -> [B, H*W]
        emb = self.timembedding(emb)

        # 5. 重塑为空间特征图 [B, 1, H, W]
        return emb.reshape(-1, 1, self.img_H, self.img_W)


# 测试函数
def test_TimeEmbedding():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = 1000
    d_model = 128
    img_H, img_W = 64, 64
    batch_size = 8

    # 创建连续时间嵌入模块
    time_embedding = TimeEmbedding(T,d_model, img_H, img_W)
    time_embedding.to(device)
    # 生成连续时间输入 (范围[0,1])
    t = torch.rand(batch_size).to(device) # 连续时间

    # 前向传播
    output = time_embedding(t)

    print("输入时间形状:", t.shape)
    print("输出嵌入形状:", output.shape)
    print("输出范围: [{:.4f}, {:.4f}]".format(
        output.min().item(), output.max().item()))

    # 梯度测试
    output.sum().backward()
    print("梯度计算成功完成!")



# 这个条件网络可以进行更改 将这个网络和UNet中的attn作类比网络
class ConditionalEmbedding(nn.Module):
    # d_model 则是嵌入向量的维度
    def __init__(self,input_shape, output_shape,C_list):
        super(ConditionalEmbedding, self).__init__()
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

def ConditionalEmbedding_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    input_shape = [4,256,256]
    output_shape = [64,64,64]
    C_list = torch.tensor([64,64,64,128,128,128,128])
    condition_map = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]).to(device)

    net = ConditionalEmbedding(input_shape,output_shape,C_list).to(device)
    output = net(condition_map)
    print(f"Output shape: {output.shape}")





class velocity_UNet(nn.Module):
    # 修改上卷积方法
    def __init__(self, T, input_shape, output_shape, C_down_list,attn_params):
        super(velocity_UNet, self).__init__()
        tdim = max(C_down_list) * 4

        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, _, _ = output_shape
        kernel_sizes = [3, 5, 7, 9]
        # 创建下采样路径（编码器）
        self.encoder = nn.ModuleList()
        in_ch = self.input_channel
        for out_ch in C_down_list:
            self.encoder.append(nn.Sequential(
                multi_scale_block2D(C_in=in_ch, C_out=out_ch, kernel_sizes=kernel_sizes, dilated_num=2),
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
        self.TimeEmbedding_up = nn.ModuleList()
        self.TimeEmbedding_down = nn.ModuleList()
        attn_channels = [
            C_down_list[2] // 4,  # 对应第4层
            C_down_list[1] // 4,  # 对应第3层
            C_down_list[0] // 4,  # 对应第2层
            C_down_list[0] // 4  # 对应第1层
        ]
        attn_factors = [8, 4, 2, 1]  # 空间尺寸缩小因子
        self.repeat_factors = [4,4,4,2]
        for ch, factor, param in zip(attn_channels, attn_factors, attn_params):
            self.attentions.append(ConditionalEmbedding(
                input_shape,
                [ch, self.input_H // factor, self.input_W // factor],
                param
            ))
            self.TimeEmbedding_up.append(TimeEmbedding(T, tdim, self.input_H // factor, self.input_W // factor))
            self.TimeEmbedding_down.append(TimeEmbedding(T, tdim, self.input_H // factor, self.input_W // factor))

        # 输出层
        now_ch = C_down_list[0] // 2
        self.tail = nn.Sequential(
            nn.GroupNorm(now_ch // 4, now_ch),
            Swish(),
            nn.Conv2d(now_ch, self.output_channel, 3, stride=1, padding=1)
        )

    def forward(self, x_t, t, condition_info):
        # 解码器路径
        attn_outputs = []
        time_outputs_up = []
        time_outputs_down = []
        x_in = torch.cat([x_t, condition_info], dim=1)
        for i in range(len(self.attentions)):
            attn_outputs.append(self.attentions[i](x_in).repeat(1, self.repeat_factors[i], 1, 1))
        for i in range(len(self.TimeEmbedding_up)):
            time_outputs_up.append(self.TimeEmbedding_up[i](t))
            time_outputs_down.append(self.TimeEmbedding_down[i](t))
        # 编码器路径
        encoder_outs = []

        for i in range(len(self.encoder)):
            x_in = self.encoder[i](x_in) + time_outputs_down[len(self.encoder) - i - 1]
            encoder_outs.append(x_in)

        # 中心处理
        down4_out = encoder_outs[-1]
        center_out = self.conv_center(down4_out)
        x = torch.cat([center_out, down4_out], dim=1)

        for i in range(len(self.decodes)):
            # 上采样卷积
            x = self.decodes[i](x)
            # 应用注意力机制
            x = x + attn_outputs[i] + time_outputs_up[i]
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



def velocity_UNet_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 8
    T = 1000
    img_H = 256
    img_W = 256
    condition_info = torch.randn(batch_size, 4, img_H, img_W).to(device)
    x_t = torch.randn(batch_size, 1, img_H, img_W).to(device)
    t = torch.rand(batch_size).to(device)
    BTM_ghost_UNet_input_shape = [condition_info.shape[1]+1, condition_info.shape[2], condition_info.shape[3]]
    BTM_ghost_UNet_output_shape = [1, condition_info.shape[2], condition_info.shape[3]]
    C_down_list = [64, 128, 256, 512]
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    net = velocity_UNet(T,BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)
    output = net(x_t, t, condition_info)
    print(f"Output shape: {output.shape}")



if __name__ == '__main__':
    # test_TimeEmbedding()
    # ConditionalEmbedding_test()
    velocity_UNet_test()



    # ConditionalEmbedding_test()
    # noise_UNet_test()
    # TEmbeding_block()
    # test_TimeEmbedding()
    # velocity_UNet_test()
    #
    # #注意力模块
    #
    # class AttnBlock(nn.Module):
    #     def __init__(self, in_ch):
    #         super().__init__()
    #         self.group_norm = nn.GroupNorm(32, in_ch)
    #         self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0) # 卷积
    #         self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
    #         self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
    #         self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
    #
    #     def forward(self, x):
    #         B, C, H, W = x.shape
    #         h = self.group_norm(x)
    #         q = self.proj_q(h)
    #         k = self.proj_k(h)
    #         v = self.proj_v(h)
    #
    #         q = q.permute(0, 2, 3, 1).view(B, H * W, C)
    #         k = k.view(B, C, H * W)
    #         w = torch.bmm(q, k) * (int(C) ** (-0.5))
    #         assert list(w.shape) == [B, H * W, H * W]
    #         w = F.softmax(w, dim=-1)
    #
    #         v = v.permute(0, 2, 3, 1).view(B, H * W, C)
    #         h = torch.bmm(w, v)
    #         assert list(h.shape) == [B, H * W, C]
    #         h = h.view(B, H, W, C).permute(0, 3, 1, 2)
    #         h = self.proj(h)
    #
    #         return x + h

    # #残差模块
    #
    # class ResBlock(nn.Module):
    #     def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
    #         super().__init__()
    #         self.block1 = nn.Sequential(
    #             nn.GroupNorm(32, in_ch),
    #             Swish(),
    #             nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
    #         )
    #         self.temb_proj = nn.Sequential(
    #             Swish(),
    #             nn.Linear(tdim, out_ch),
    #         )
    #         self.cond_proj = nn.Sequential(
    #             Swish(),
    #             nn.Linear(tdim, out_ch),
    #         )
    #         self.block2 = nn.Sequential(
    #             nn.GroupNorm(32, out_ch),
    #             Swish(),
    #             nn.Dropout(dropout),
    #             nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
    #         )
    #         if in_ch != out_ch:
    #             self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
    #         else:
    #             self.shortcut = nn.Identity()
    #         if attn:
    #             self.attn = AttnBlock(out_ch)
    #         else:
    #             self.attn = nn.Identity()
    #
    #
    #     def forward(self, x, temb, labels):
    #         h = self.block1(x)
    #         h += self.temb_proj(temb)[:, :, None, None]
    #         h += self.cond_proj(labels)[:, :, None, None]
    #         h = self.block2(h)
    #
    #         h = h + self.shortcut(x)
    #         h = self.attn(h)
    #         return h
    #
    #
    # class UNet(nn.Module):
    #     def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
    #         super().__init__()
    #         tdim = ch * 4
    #         self.time_embedding = TimeEmbedding(T, ch, tdim)
    #         self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
    #         self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
    #         self.downblocks = nn.ModuleList()
    #         chs = [ch]  # record output channel when dowmsample for upsample
    #         now_ch = ch
    #         for i, mult in enumerate(ch_mult):
    #             out_ch = ch * mult
    #             for _ in range(num_res_blocks):
    #                 self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
    #                 now_ch = out_ch
    #                 chs.append(now_ch)
    #             if i != len(ch_mult) - 1:
    #                 self.downblocks.append(DownSample(now_ch))
    #                 chs.append(now_ch)
    #
    #         self.middleblocks = nn.ModuleList([
    #             ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
    #             ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
    #         ])
    #
    #         self.upblocks = nn.ModuleList()
    #         for i, mult in reversed(list(enumerate(ch_mult))):
    #             out_ch = ch * mult
    #             for _ in range(num_res_blocks + 1):
    #                 self.upblocks.append \
    #                     (ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
    #                 now_ch = out_ch
    #             if i != 0:
    #                 self.upblocks.append(UpSample(now_ch))
    #         assert len(chs) == 0
    #
    #         self.tail = nn.Sequential(
    #             nn.GroupNorm(32, now_ch),
    #             Swish(),
    #             nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
    #         )
    #
    #
    #     def forward(self, x, t, labels):
    #         # Timestep embedding
    #         temb = self.time_embedding(t)
    #         cemb = self.cond_embedding(labels)
    #         # Downsampling
    #         h = self.head(x)
    #         hs = [h]
    #         for layer in self.downblocks:
    #             h = layer(h, temb, cemb)
    #             hs.append(h)
    #         # Middle
    #         for layer in self.middleblocks:
    #             h = layer(h, temb, cemb)
    #         # Upsampling
    #         for layer in self.upblocks:
    #             if isinstance(layer, ResBlock):
    #                 h = torch.cat([h, hs.pop()], dim=1)
    #             h = layer(h, temb, cemb)
    #         h = self.tail(h)
    #
    #         assert len(hs) == 0
    #         return h
    #
    # def UNet_example():
    #     batch_size = 8
    #     model = UNet(
    #         T=1000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
    #         num_res_blocks=2, dropout=0.1)
    #     x = torch.randn(batch_size, 3, 32, 32)
    #     t = torch.randint(1000, size=[batch_size])
    #     labels = torch.randint(10, size=[batch_size])
    #     # resB = ResBlock(128, 256, 64, 0.1)
    #     # x = torch.randn(batch_size, 128, 32, 32)
    #     # t = torch.randn(batch_size, 64)
    #     # labels = torch.randn(batch_size, 64)
    #     # y = resB(x, t, labels)
    #     y = model(x, t, labels)
    #     print(y.shape)

    # def attan_block():
    #
    #     # 创建模拟输入 (2个样本, 64通道, 32x32特征图)
    #     x = torch.randn(2, 64, 32, 32)  # shape: [B, C, H, W]
    #     attn_block = AttnBlock(in_ch=64)
    #     output = attn_block(x)
    #
    #     print("Input shape:", x.shape)  # torch.Size([2, 64, 32, 32])
    #     print("Output shape:", output.shape)  # torch.Size([2, 64, 32, 32])

