
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from model.sub_block.mid_conv import *

# def drop_connect(x, drop_ratio):
#     keep_ratio = 1.0 - drop_ratio
#     mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
#     mask.bernoulli_(p=keep_ratio)
#     x.div_(keep_ratio  )# 4. 缩放输入以保持期望值不变
#     x.mul_(mask  )# 5. 应用掩码 - 随机将整个特征图置零
#     return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 这个暂时不动
class TimeEmbedding(nn.Module):
    # d_model 是频率嵌入维度
    def __init__(self, T, d_model, img_H, img_W):
        self.img_H, self.img_W = img_H, img_W
        assert d_model % 2 == 0
        super().__init__()
        # 将math.log(10000)分成 d_model/2 份
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        # 映射到 (0.0001, 1]
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        #逐元素乘法 (T, d_model/2) T是频率步数 emb 则是频率
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        # 得到两个正交分量矩阵  (T, d_model//2,2)
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        # 先放所有 sin，再放所有 cos
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            # 时间编码过程  并且 freeze=False 是可以梯度下降的 为 nn.Embedding.from_pretrained
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, self.img_H),
            Swish(),
            nn.Linear(self.img_H, self.img_H*self.img_W),
        )

    def forward(self, t):
        emb = self.timembedding(t).reshape(-1,1, self.img_H, self.img_W)

        return emb

def TEmbeding_block():
    # 测试参数
    T = 1000  # 时间步总数
    d_model = 128  # 嵌入维度
    img_H,img_W = 256,256  # 输出维度
    batch_size = 8  # 批大小
    # 创建测试实例
    time_embedding = TimeEmbedding(T, d_model, img_H, img_W)
    # 生成随机时间步 (范围在 0 到 T-1 之间)
    t = torch.randint(0, T, (batch_size,))
    # 前向传播
    output = time_embedding(t)
    # 打印结果
    print("输入时间步形状:", t.shape)
    print("输出嵌入形状:", output.shape)
    print("\n前几个时间步的输出示例:")
    # print(output[:3])  # 打印前3个样本的输出

    # 验证梯度计算
    output.sum().backward()
    print("\n梯度计算成功完成!")



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
        layers.append(Res_Inception_ghost2D(C_in=self.input_channel, C_out=C_list[0], kernel_sizes=[1, 3, 5, 7], dilated_num=1))
        for i in range(len(C_list) - 1):
            layers.append(Res_Inception_ghost2D(C_in=C_list[i], C_out=C_list[i + 1], kernel_sizes=[1, 3, 5, 7], dilated_num=1))
        self.encoder = nn.Sequential(*layers)
        self.gelu = nn.GELU()

    def forward(self, condition_map):

        resized_map = F.interpolate(condition_map, size=(self.output_H, self.output_W), mode='bilinear', align_corners=True)
        out_map = self.encoder(resized_map)

        return out_map

def ConditionalEmbedding_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    input_shape = [4,256,256]
    output_shape = [64,64,64]
    C_list = torch.tensor([64,64,64,128,128,128,128])
    condition_map = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]).to(device)

    net = ConditionalEmbedding(input_shape,output_shape,C_list).to(device)
    output = net(condition_map)
    print(f"Output shape: {output.shape}")


# # 下采样和上采样可以进行平替
# # 上下采样可以进行更改
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        x = self.c1(x) + self.c2(x)
        return x


# 上下采样可以进行更改
class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


class noise_UNet(nn.Module):
    # 修改上卷积方法
    def __init__(self, T, input_shape, output_shape, C_down_list,attn_params):
        super(noise_UNet, self).__init__()
        tdim = max(C_down_list) * 4

        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, _, _ = output_shape
        kernel_sizes = [3, 5, 7, 9]
        # 创建下采样路径（编码器）
        self.encoder = nn.ModuleList()
        in_ch = self.input_channel
        for out_ch in C_down_list:
            self.encoder.append(nn.Sequential(
                Res_Inception_ghost2D(C_in=in_ch, C_out=out_ch, kernel_sizes=kernel_sizes, dilated_num=2),
                Conv_DownSampling2D(out_ch)
            ))
            in_ch = out_ch

        # 中心卷积层
        self.conv_center = Res_Fractal_inception2D(input_channel=C_down_list[-1],output_channel=C_down_list[-1])

        # 创建上采样路径（解码器）
        self.decodes = nn.ModuleList()
        for i in range(len(C_down_list), 0, -1):
            i = i -1
            self.decodes.append(
                nn.Sequential(
                    Res_Inception_ghost2D(C_in=C_down_list[i] + C_down_list[i],C_out=C_down_list[i],kernel_sizes=kernel_sizes,dilated_num=1),
                    ConvTranspose_UpSam(C_down_list[i])
                )
            )
        # 创建注意力模块
        self.attentions = nn.ModuleList()
        self.TimeEmbeddings = nn.ModuleList()
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
            self.TimeEmbeddings.append(TimeEmbedding(T, tdim, self.input_H // factor, self.input_W // factor))

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
        time_outputs = []
        x_in = torch.cat([x_t, condition_info], dim=1)
        for i in range(len(self.attentions)):
            attn_outputs.append(self.attentions[i](x_in).repeat(1, self.repeat_factors[i], 1, 1))
        for i in range(len(self.TimeEmbeddings)):
            time_outputs.append(self.TimeEmbeddings[i](t))
        # 编码器路径
        encoder_outs = []
        for layer in self.encoder:
            x_in = layer(x_in)
            encoder_outs.append(x_in)

        # 中心处理
        down4_out = encoder_outs[-1]
        center_out = self.conv_center(down4_out)
        x = torch.cat([center_out, down4_out], dim=1)

        for i in range(len(self.decodes)):
            # 上采样卷积
            x = self.decodes[i](x)
            # 应用注意力机制
            x = (x + attn_outputs[i])*time_outputs[i]
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


def noise_UNet_test():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 8
    T = 1000
    img_H = 256
    img_W = 256
    condition_info = torch.randn(batch_size, 4, img_H, img_W).to(device)
    x_t = torch.randn(batch_size, 1, img_H, img_W).to(device)
    t = torch.randint(1000, size=[batch_size]).to(device)
    BTM_ghost_UNet_input_shape = [condition_info.shape[1]+1, condition_info.shape[2], condition_info.shape[3]]
    BTM_ghost_UNet_output_shape = [1, condition_info.shape[2], condition_info.shape[3]]
    C_down_list = [64, 128, 256, 512]
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    net = noise_UNet(T,BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)
    output = net(x_t, t, condition_info)
    print(f"Output shape: {output.shape}")



if __name__ == '__main__':
    # ConditionalEmbedding_test()
    # noise_UNet_test()
    # TEmbeding_block()
    noise_UNet_test()
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

