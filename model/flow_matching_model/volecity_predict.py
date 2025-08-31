import torch
from torch import nn
from torch.nn import functional as F
from model.sub_block.mid_conv import inception_ghost_sum
from model.sub_block.low_conv import BasicNormConv
import time

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
    def __init__(self,input_shape, output_shape,C_list,kernel_list,dilated_list):
        super(ConditionalEmbedding, self).__init__()
        self.input_channel , self.input_H , self.input_W = input_shape
        self.output_channel , self.output_H , self.output_W = output_shape

        C_list = torch.cat((C_list, torch.tensor([self.output_channel], dtype=torch.int32)))
        # 构建编码器
        layers = []

        layers.append(inception_ghost_sum(C_in=self.input_channel, C_out=C_list[0], kernel_list=kernel_list, dilated_list=dilated_list,drop_out=0))
        for i in range(len(C_list) - 1):
            layers.append(inception_ghost_sum(C_in=C_list[i], C_out=C_list[i + 1], kernel_list=kernel_list, dilated_list=dilated_list,drop_out=0))
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
    kernel_list = [3, 5, 7, 9]
    dilated_list = [1, 1, 1, 1]
    condition_map = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]).to(device)

    net = ConditionalEmbedding(input_shape = input_shape,
                               output_shape = output_shape,
                               C_list = C_list,
                               kernel_list = kernel_list,
                               dilated_list = dilated_list).to(device)
    output = net(condition_map)
    print(f"Input shape: {condition_map.shape}")
    print(f"Output shape: {output.shape}")

#标准 注意力模块

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





def attan_block_test():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


    # 清空GPU缓存并记录初始显存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB

    print(f"初始显存占用: {initial_memory:.2f} MB")
    C_in = 512
    # 创建模拟输入
    x = torch.randn(8, C_in, 32, 32).to(device)
    input_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory
    print(f"输入张量显存占用: {input_memory:.2f} MB")

    # 创建注意力块
    attn_block = AttnBlock(in_ch=C_in).to(device)
    model_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory - input_memory
    print(f"模型参数显存占用: {model_memory:.2f} MB")

    # 前向传播
    output = attn_block(x)

    forward_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory - input_memory - model_memory
    print(f"前向传播中间变量显存占用: {forward_memory:.2f} MB")

    total_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
    print(f"总显存占用: {total_memory:.2f} MB")

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # 峰值显存使用
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f"峰值显存使用: {peak_memory:.2f} MB")

    return output

# COT模块
class CoTAttention(nn.Module):

    def __init__(self, C_in, kernel_size):
        super().__init__()
        factor = 4
        C_mid = 2 * C_in // factor
        self.C_in = C_in
        self.kernel_size = kernel_size
        self.key_embed =BasicNormConv(C_in=C_in, C_out=C_in, kernel_size=kernel_size, gelu=True, norm=True)
        self.value_embed = BasicNormConv(C_in=C_in, C_out=C_in, kernel_size=1, gelu=False, norm=True)
        self.attention_embed = nn.Sequential(BasicNormConv(C_in=2 * C_in, C_out=C_mid, kernel_size=1, gelu=True, norm=True),
                                             BasicNormConv(C_in=C_mid, C_out=kernel_size * kernel_size * C_in, kernel_size=1,gelu=False, norm=False))

    def forward(self, x):
        bs, c, h, w = x.shape

        k1 = self.key_embed(x)  # 编码静态上下文信息key,表示为k1: (B,C,H,W) --> (B,C,H,W)
        v = self.value_embed(x)  # 编码value矩阵: (B,C,H,W) --> (B,C,H,W)

        # 对value进行填充以便进行unfold操作
        padding = (self.kernel_size - 1) // 2
        v_padded = F.pad(v, (padding, padding, padding, padding), mode='constant', value=0)
        v_unfold = F.unfold(v_padded, kernel_size=self.kernel_size)  # 形状: (B, C*k*k, H*W)
        v_unfold = v_unfold.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)  # 重塑为 (B, C, k*k, H, W)

        y = torch.cat([k1, x], dim=1)  # 将上下文信息key和query在通道上进行拼接: (B,2C,H,W)

        att = self.attention_embed(y)  # 通过两个连续的1×1卷积操作: (B,2C,H,W)-->(B,D,H,W)-->(B,C×k×k,H,W)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)  # (B,C×k×k,H,W) --> (B,C,k×k,H,W)

        k2 = (v_unfold * att).sum(dim=2)  # 形状: (B, C, H, W)

        return k1 + k2

def COT_test():
    print("CoT_test")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 清空GPU缓存并记录初始显存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
    print(f"初始显存占用: {initial_memory:.2f} MB")
    C_in = 128
    # 创建输入张量
    input = torch.randn(16, C_in, 128, 128).to(device)
    input_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory
    print(f"输入张量显存占用: {input_memory:.2f} MB")

    # 创建模型
    Model = CoTAttention(C_in=C_in, kernel_size=7).to(device)
    model_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory - input_memory
    print(f"模型参数显存占用: {model_memory:.2f} MB")

    # 前向传播
    start_time = time.time()
    output = Model(input)
    unfold_time = time.time() - start_time
    print(f"Unfold实现时间: {unfold_time:.4f} 秒")
    forward_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 - initial_memory - input_memory - model_memory
    print(f"前向传播中间变量显存占用: {forward_memory:.2f} MB")

    # 统计信息
    total_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
    print(f"总显存占用: {total_memory:.2f} MB")

    # 峰值显存使用
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f"峰值显存使用: {peak_memory:.2f} MB")

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)

    return output

# LSKNet

class LSKmodule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3) # 应用padding使输入输出shape保持一致
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # x: (B,C,H,W)

        # (k:卷积核尺寸,d:膨胀率)  将(23,1)分解为: (5,1) and (7,3)
        attn1 = self.conv0(x)     # 应用第一个卷积层: (B,C,H,W)--> (B,C,H,W)
        attn2 = self.convl(attn1) # 应用第二个卷积层: (B,C,H,W)--> (B,C,H,W)

        attn1 = self.conv0_s(attn1)  # 应用1×1Conv建模通道间相关性,并将通道降维到C/2: (B,C,H,W)--> (B,C/2,H,W)   注意:如果分解为了三个卷积层,那就将通道C降维到C/3, 以便于在Concat的时候能恢复到原通道数量
        attn2 = self.conv1_s(attn2)  # 应用1×1Conv建模通道间相关性,并将通道降维到C/2: (B,C,H,W)--> (B,C/2,H,W)   # 原文中并没有提出需要降维,应该是为了提高计算效率选择降维了

        attn = torch.cat([attn1, attn2], dim=1)  # 将多个不同尺度的特征图在通道上进行拼接,恢复原通道数量: (B,C,H,W)
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # 应用全局平均池化: (B,C,H,W)-->(B,1,H,W)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # 应用全局最大池化: (B,C,H,W)-->(B,1,H,W)
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 将平均池化和最大池化特征进行拼接: (B,2,H,W)
        sig = self.conv_squeeze(agg).sigmoid()  # 将2个通道映射为N个通道, N是尺度的个数, 并通过sigmoid函数得到每个尺度对应的权重表示: (B,N,H,W), 在这里N==2
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1) # 对多个尺度的信息进行加权求和: (B,C/2,H,W)
        attn = self.conv_m(attn) # 将通道恢复为原通道数量: (B,C/2,H,W)-->(B,C,H,W)
        return x * attn  # 最后与输入特征执行逐元素乘法

def LSK_test():
    input = torch.randn(1, 512, 7, 7)
    Model = LSKmodule(dim=512)
    output = Model(input)
    print(output.shape)



# class velocity_UNet(nn.Module):
#     # 修改上卷积方法
#     def __init__(self, T, input_shape, output_shape, C_down_list,attn_params):
#         super(velocity_UNet, self).__init__()
#         tdim = max(C_down_list) * 4
#
#         self.input_channel, self.input_H, self.input_W = input_shape
#         self.output_channel, _, _ = output_shape
#         kernel_sizes = [3, 5, 7, 9]
#         convDownKernel_list = [3, 5, 7]
#         convUpKernel_list = [3, 5, 7]
#         # 创建下采样路径（编码器）
#         self.encoder = nn.ModuleList()
#         in_ch = self.input_channel
#         for out_ch in C_down_list:
#             self.encoder.append(nn.Sequential(
#                 multi_scale_block2D(C_in=in_ch, C_out=out_ch, kernel_sizes=kernel_sizes, dilated_num=2),
#                 multiScaleConvDown(out_ch,convDownKernel_list)
#             ))
#             in_ch = out_ch
#
#         # 中心卷积层
#         self.conv_center = Fractal_multi_scale2D(input_channel=C_down_list[-1],output_channel=C_down_list[-1])
#
#         # 创建上采样路径（解码器）
#         self.decodes = nn.ModuleList()
#         for i in range(len(C_down_list), 0, -1):
#             i = i -1
#             self.decodes.append(
#                 nn.Sequential(
#                     multi_scale_block2D(C_in=C_down_list[i] + C_down_list[i],C_out=C_down_list[i],kernel_sizes=kernel_sizes,dilated_num=1),
#                     MultiScaleUpSample(C_down_list[i],convUpKernel_list)
#                 )
#             )
#         # 创建注意力模块
#         self.attentions = nn.ModuleList()
#         self.TimeEmbeddings = nn.ModuleList()
#         attn_channels = [
#             C_down_list[2] // 4,  # 对应第4层
#             C_down_list[1] // 4,  # 对应第3层
#             C_down_list[0] // 4,  # 对应第2层
#             C_down_list[0] // 4  # 对应第1层
#         ]
#         attn_factors = [8, 4, 2, 1]  # 空间尺寸缩小因子
#         self.repeat_factors = [4,4,4,2]
#         for ch, factor, param in zip(attn_channels, attn_factors, attn_params):
#             self.attentions.append(ConditionalEmbedding(
#                 input_shape,
#                 [ch, self.input_H // factor, self.input_W // factor],
#                 param
#             ))
#             self.TimeEmbeddings.append(TimeEmbedding(T, tdim, self.input_H // factor, self.input_W // factor))
#
#         # 输出层
#         now_ch = C_down_list[0] // 2
#         self.tail = nn.Sequential(
#             nn.GroupNorm(now_ch // 4, now_ch),
#             Swish(),
#             nn.Conv2d(now_ch, self.output_channel, 3, stride=1, padding=1)
#         )
#
#     def forward(self, x_t, t, condition_info):
#         # 解码器路径
#         attn_outputs = []
#         time_outputs = []
#         x_in = torch.cat([x_t, condition_info], dim=1)
#         for i in range(len(self.attentions)):
#             attn_outputs.append(self.attentions[i](x_in).repeat(1, self.repeat_factors[i], 1, 1))
#         for i in range(len(self.TimeEmbeddings)):
#             time_outputs.append(self.TimeEmbeddings[i](t))
#         # 编码器路径
#         encoder_outs = []
#         for layer in self.encoder:
#             x_in = layer(x_in)
#             encoder_outs.append(x_in)
#
#         # 中心处理
#         down4_out = encoder_outs[-1]
#         center_out = self.conv_center(down4_out)
#         x = torch.cat([center_out, down4_out], dim=1)
#
#         for i in range(len(self.decodes)):
#             # 上采样卷积
#             x = self.decodes[i](x)
#             # 应用注意力机制
#             x = x + attn_outputs[i] + time_outputs[i]
#             # 跳跃连接（拼接编码器特征）
#             skip_idx = len(encoder_outs) - 2 - i
#             if skip_idx >= 0:
#                 x = torch.cat([x, encoder_outs[skip_idx]], dim=1)
#
#         # 最终输出层
#         return self.tail(x * attn_outputs[-1])
#
#     def load_weights(self, checkpoint_path):
#         """加载预训练权重"""
#         self.load_state_dict(torch.load(checkpoint_path, weights_only=True))
#         print(f"Loaded weights from {checkpoint_path}")
#
#
#
# def velocity_UNet_test():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
#     batch_size = 8
#     T = 1000
#     img_H = 256
#     img_W = 256
#     condition_info = torch.randn(batch_size, 4, img_H, img_W).to(device)
#     x_t = torch.randn(batch_size, 1, img_H, img_W).to(device)
#     t = torch.rand(batch_size).to(device)
#     BTM_ghost_UNet_input_shape = [condition_info.shape[1]+1, condition_info.shape[2], condition_info.shape[3]]
#     BTM_ghost_UNet_output_shape = [1, condition_info.shape[2], condition_info.shape[3]]
#     C_down_list = [64, 128, 256, 512]
#     C_list_attn = torch.tensor([64, 64, 128, 128, 128])
#     attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
#     net = velocity_UNet(T,BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)
#     output = net(x_t, t, condition_info)
#     print(f"Output shape: {output.shape}")



if __name__ == '__main__':
    # test_TimeEmbedding()

    # velocity_UNet_test()
    # ConditionalEmbedding_test()
    LSK_test()


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

