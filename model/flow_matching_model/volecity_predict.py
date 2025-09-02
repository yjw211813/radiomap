import torch
from torch import nn
from torch.nn import functional as F
from model.sub_block.mid_conv import inception_ghost_sum
from model.sub_block.low_conv import BasicNormConv,multiScaleConvDown,multiScaleUpSample
from model.sub_block.top_conv import fractal_conv,inception_sum,MSAA_space_atten
import time
from model.sub_block.statistic_tools import gpu_statistic

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# input   t (torch.Tensor): 时间向量，形状为 [B]
# output   torch.Tensor: 时间嵌入特征图，形状为 [B, 1, H, W]
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

# 这个条件网络可以进行更改 将这个网络和UNet中的attn作类比网络
# 输入 input_shape
# 输出 output_shape

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

#标准 注意力模块
# 输入 [B, C, H, W]
# 输出 [B, C, H, W]
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
    print("attan_block_test")

    # 清空GPU缓存并记录初始显存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB

    print(f"初始显存占用: {initial_memory:.2f} MB")
    C_in = 256
    img_size = 128
    # 创建模拟输入
    x = torch.randn(16, C_in, img_size, img_size).to(device)
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
# 输入 [B, C, H, W]
# 输出 [B, C, H, W]
class CoTAttention(nn.Module):

    def __init__(self, C_in, kernel_size):
        super(CoTAttention,self).__init__()
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

# LSKNet
# 输入 [B, C, H, W]
# 输出 [B, C, H, W]
class LSKNet(nn.Module):
    def __init__(self, C_in, kernel_mid, kernel_list, dilated_list, drop_out=0,factor = 2):
        super(LSKNet,self).__init__()
        if C_in % len(kernel_list) != 0:
            raise ValueError(f"C_in ({C_in}) must be divisible by len(kernel_list)")
        self.sub_Cout = int(C_in / len(kernel_list) * factor)

        self.conv_list = nn.ModuleList()
        self.conv_trans = nn.ModuleList()
        for i in range(len(kernel_list)):
            self.conv_list.append(BasicNormConv(C_in=C_in, C_out=C_in,
                                                kernel_size=kernel_list[i],
                                                dilation=dilated_list[i],
                                                norm=False,gelu=False,
                                                dropout_rate=drop_out, groups=C_in))
            self.conv_trans.append(BasicNormConv(C_in=C_in, C_out=self.sub_Cout,
                                                 kernel_size=kernel_list[i],
                                                 norm=False, gelu=False,
                                                 dropout_rate=drop_out))

        self.conv_squeeze = BasicNormConv(C_in=2, C_out=len(kernel_list),
                                          norm=False, gelu=False,
                                          kernel_size=kernel_mid, dropout_rate=drop_out)
        self.conv_m = BasicNormConv(C_in=self.sub_Cout, C_out=C_in,
                                    norm=False, gelu=False,
                                    kernel_size=1, dropout_rate=drop_out)

    def forward(self, x):
        # x: (B,C,H,W)
        attn_list = []
        kernel_num = len(self.conv_list)
        for i in range(kernel_num):
            if i == 0:
                attn_list.append(self.conv_list[i](x))
            else:
                attn_list.append(self.conv_list[i](attn_list[i - 1]))

        for i in range(kernel_num):
            attn_list[i] = self.conv_trans[i](attn_list[i])

        attn = torch.cat(attn_list, dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)  # 应用全局平均池化: (B,C,H,W)-->(B,1,H,W)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # 应用全局最大池化: (B,C,H,W)-->(B,1,H,W)
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 将平均池化和最大池化特征进行拼接: (B,2,H,W)
        sig = self.conv_squeeze(agg).sigmoid()  # 将2个通道映射为N个通道, N是尺度的个数, 并通过sigmoid函数得到每个尺度对应的权重表示: (B,N,H,W), 在这里N==2

        # 修正这里的语法错误
        weighted_attn = 0
        for i in range(kernel_num):
            weighted_attn += attn_list[i] * sig[:, i:i + 1, :, :]  # 使用广播机制

        weighted_attn = self.conv_m(weighted_attn)
        return x * weighted_attn


def LSK_test():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    C_in = 128
    kernel_mid = 7
    kernel_list = [5,7,5,5]
    dilated_list = [1,3,3,3]
    img_size = 128
    x = torch.randn(16, C_in, img_size, img_size)
    model = LSKNet(C_in, kernel_mid, kernel_list, dilated_list)
    get_gpu_info.print_gpu_memory("LSK_test GPU info",x,model)

#残差模块
# 输入 x = [B, C_in, H, W] temb = [B, 1, H, W]
# 输出 out = [B, C_out, H, W]
class ResConv(nn.Module):
    def __init__(self, C_in, C_out, conv_kernels,conv_dilats,drop_out=0.05, attn="Attn", LSK_kernels = [5,7,5,5], LSK_dilats = [1,3,3,3], LSK_mid_kernel = 7):
        super().__init__()
        self.conv_begin = inception_ghost_sum(C_in, C_out, conv_kernels, conv_dilats, drop_out)
        self.conv_end = inception_ghost_sum(C_out, C_out, conv_kernels, conv_dilats, drop_out)
        if C_in != C_out:
            self.shortcut = BasicNormConv(C_in = C_in, C_out = C_out, kernel_size = 1,gelu=False,norm = False)
        else:
            self.shortcut = nn.Identity()
        if attn == "LSKNet":
            self.attn = LSKNet(C_out, kernel_mid = LSK_mid_kernel, kernel_list = LSK_kernels, dilated_list = LSK_dilats)
        else:
            self.attn = AttnBlock(C_out)

    def forward(self, x, temb):
        out =self.conv_end( self.conv_begin(x) + temb)
        out = self.attn(out + self.shortcut(x))
        return out

def ResConv_test():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    C_in = 4
    C_out = 64

    conv_kernels = [3,5,7,9]
    conv_dilats = [1,1,1,1]
    LSK_kernels = [5,7,5,5]
    LSK_dilats = [1,3,3,3]
    LSK_mid_kernel = 7
    img_size = 256
    x = torch.randn(16, C_in, img_size, img_size)
    temb = torch.randn(16, 1, img_size, img_size)
    model = ResConv(C_in = C_in,C_out = C_out,
                    conv_kernels = conv_kernels,
                    conv_dilats = conv_dilats,
                    attn="LSKNet",
                    LSK_kernels = LSK_kernels,
                    LSK_dilats = LSK_dilats,
                    LSK_mid_kernel = LSK_mid_kernel)
    get_gpu_info.print_gpu_memory("ResConv LSKNet GPU info",x,model,temb = temb)
    img_size = 64
    C_in = 256
    C_out = 512
    x = torch.randn(16, C_in, img_size, img_size)
    temb = torch.randn(16, 1, img_size, img_size)
    model = ResConv(C_in=C_in, C_out=C_out,
                    conv_kernels=conv_kernels,
                    conv_dilats=conv_dilats)
    get_gpu_info.print_gpu_memory("ResConv Attn GPU info",x,model,temb = temb)


# resconv 参数部分 卷积 conv_kernels，conv_dilats  注意力 LSK_kernels，LSK_dilats，LSK_mid_kernel
# 下采样 多尺度 convDownKernels 上采样 多尺度 convUpKernels
# 应该有的通道主干 C_list =[64, 128, 256, 512]
# input_shape = [B, C, H, W] output_shape = [B, 1, H, W]
#     conv_kernels = [3,5,7,9]
#     conv_dilats = [1,1,1,1]
#     LSK_kernels = [5,7,5,5]
#     LSK_dilats = [1,3,3,3]
#     LSK_mid_kernel = 7
#     T 为总步长
# convDownKernels = [3, 5, 7] convUpKernels = [3,5,7]
class velocity_UNet(nn.Module):
    def __init__(self, net_info_dict):
        """
        速度场UNet网络模型
        参数:
            net_info_dict (dict): 包含网络配置信息的字典
        """
        super(velocity_UNet, self).__init__()

        # 初始化输入参数
        self.input_channel = net_info_dict["in_shape"][1]  # 输入通道数
        self.input_H = net_info_dict["in_shape"][2]  # 输入高度
        self.input_W = net_info_dict["in_shape"][3]  # 输入宽度

        # 编码器部分
        self.encoder_conv_list = nn.ModuleList()  # 编码器卷积层列表
        self.encoder_down_list = nn.ModuleList()  # 编码器下采样层列表
        
        C_in = self.input_channel
        for i, C_out in enumerate(net_info_dict["C_list"]):
            # 添加编码器卷积层
            self.encoder_conv_list.append(
                ResConv(
                    C_in=C_in, 
                    C_out=C_out,
                    conv_kernels=net_info_dict["conv_kernels"],
                    conv_dilats=net_info_dict["conv_dilats"],
                    attn=net_info_dict["attn_list"][i],
                    LSK_kernels=net_info_dict["LSK_kernels"],
                    LSK_dilats=net_info_dict["LSK_dilats"],
                    LSK_mid_kernel=net_info_dict["LSK_mid_kernel"]
                )
            )
            
            # 添加编码器下采样层
            self.encoder_down_list.append(
                multiScaleConvDown(
                    C=C_out, 
                    kernel_list=net_info_dict["encoderDownKernels"]
                )
            )
            
            C_in = C_out  # 更新输入通道数为当前输出通道数


        # 中心卷积层
        self.conv_center = fractal_conv(
            C_in=C_out,  # 使用最后一个编码器层的输出通道数
            C_out=C_out,
            kernel_list=net_info_dict["fra_kernels"],
            dilated_list=net_info_dict["fra_dilates"],
            inception_module=inception_sum
        )
        self.atten_center = AttnBlock(C_out)
        # 中间注意力模块列表
        self.mid_attn_list = nn.ModuleList()
        # 为每个编码器层输出和输入添加注意力模块
        for i in range(len(net_info_dict["C_list"]) + 1):
            # 确定输入通道数：第一层使用原始输入通道数，后续使用对应编码器层输出通道数
            attn_C_in = net_info_dict["in_shape"][1] if i == 0 else net_info_dict["C_list"][i - 1]
            
            self.mid_attn_list.append(
                MSAA_space_atten(
                    C_in=attn_C_in,
                    space_pool_kernel=net_info_dict["MSAA_pool_kernel"],
                    kernel_list=net_info_dict["MSAA_kernels"],
                    dilated_list=net_info_dict["MSAA_dilats"]
                )
            )



        # 解码器部分
        self.decode_conv_list = nn.ModuleList()  # 解码器卷积层列表
        self.decode_up_list = nn.ModuleList()   # 解码器上采样层列表
        
        # 从最深到最浅构建解码器
        for i in range(len(net_info_dict["C_list"]) - 1, -1, -1):
            # 解码器输入是上采样输出与跳跃连接的拼接
            C_in = net_info_dict["C_list"][i] * 2  # 通道数翻倍
            C_out = net_info_dict["C_list"][i] // 2     # 输出通道数与对应编码器层相同

            self.decode_up_list.append(
                multiScaleUpSample(
                    C_in=net_info_dict["C_list"][i],
                    kernel_list=net_info_dict["decoderUpKernels"]
                )
            )
            self.decode_conv_list.append(
                ResConv(
                    C_in=C_in, 
                    C_out=C_out,
                    conv_kernels=net_info_dict["conv_kernels"],
                    conv_dilats=net_info_dict["conv_dilats"],
                    attn=net_info_dict["attn_list"][i], 
                    LSK_kernels=net_info_dict["LSK_kernels"],
                    LSK_dilats=net_info_dict["LSK_dilats"],
                    LSK_mid_kernel=net_info_dict["LSK_mid_kernel"]
                )
            )
            


        # 时间嵌入模块
        self.Time_encode_Embeddings = nn.ModuleList()  # 编码器时间嵌入
        self.Time_decode_Embeddings = nn.ModuleList()  # 解码器时间嵌入
        
        # 空间尺寸缩小因子列表
        attn_factors = [8, 4, 2, 1]
        
        # 为每个尺度创建时间嵌入模块
        for factor in attn_factors:
            # 计算当前尺度的特征图尺寸
            H_scale = self.input_H // factor
            W_scale = self.input_W // factor
            
            self.Time_encode_Embeddings.append(
                TimeEmbedding(
                    T = net_info_dict["T"], 
                    d_model = net_info_dict["tdim"], 
                    img_H = H_scale, 
                    img_W = W_scale
                )
            )
            
            self.Time_decode_Embeddings.append(
                TimeEmbedding(
                    T = net_info_dict["T"], 
                    d_model = net_info_dict["tdim"], 
                    img_H = H_scale, 
                    img_W = W_scale
                )
            )


        # 输出层
        # 计算最终拼接后的通道数
        now_ch = net_info_dict["C_list"][0] // 2 + net_info_dict["in_shape"][1]
        
        self.tail = nn.Sequential(
            nn.BatchNorm2d(now_ch),  # 分组归一化
            nn.Conv2d(
                now_ch, 
                net_info_dict["out_shape"][1], 
                net_info_dict["tail_kernel"], 
                padding=net_info_dict["tail_kernel"] // 2  # 保持尺寸不变的填充
            )
        )

    def forward(self, x_t, t, condition_info):
        """
        前向传播
        
        参数:
            x_t (Tensor): 输入张量，形状为 [B, C_out, H, W]
            t (Tensor): 时间向量，形状为 [B]
            condition_info (Tensor): 条件信息张量，形状为 [B, C_condition, H, W]
            
        返回:
            Tensor: 输出速度场，形状为 [B, C_out, H, W]
        """
        # 预处理输入：拼接输入和条件信息
        conv_num = len(self.encoder_conv_list)
        x_in = torch.cat([x_t, condition_info], dim=1)
        
        # 计算所有尺度的时间嵌入
        time_encode_embs = [te(t) for te in self.Time_encode_Embeddings]
        time_decode_embs = [td(t) for td in self.Time_decode_Embeddings]

        # 编码器路径
        encoder_outs = [x_in]  # 保存各层输出用于跳跃连接
        
        x = x_in
        for i, (conv, down) in enumerate(zip(self.encoder_conv_list, self.encoder_down_list)):
            # 应用卷积层并加入时间嵌入
            x = conv(x, time_encode_embs[conv_num - i-1])
            # 保存输出用于跳跃连接
            encoder_outs.append(x)
            # 下采样
            x = down(x)


        # 中心处理
        x = self.atten_center(self.conv_center(x))

        # 解码器路径
        for i, (conv, up) in enumerate(zip(self.decode_conv_list, self.decode_up_list)):
            # 上采样
            x = up(x)
            # 与对应编码器层输出拼接（通过注意力模块处理）
            skip_connection = self.mid_attn_list[conv_num - i](encoder_outs[-(i+1)])
            x = torch.cat([x, skip_connection], dim=1)
            # 应用卷积层并加入时间嵌入
            x = conv(x, time_decode_embs[i])
        
        # 最终输出处理：与输入层注意力处理结果拼接
        final_skip = self.mid_attn_list[0](encoder_outs[0])
        x = torch.cat([x, final_skip], dim=1)

        return self.tail(x)
    
def velocity_UNet_test():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    batch_size = 16
    T = 50
    img_H = 256
    img_W = 256

    net_info_dict = {
        "T": T,
        "in_shape": [batch_size, 6, img_H, img_W],
        "out_shape": [batch_size, 1, img_H, img_W],
        "C_list": [64, 128, 256, 512],
        "attn_list": ["LSKNet", "LSKNet", "Attn", "Attn"],
        "conv_kernels": [3, 5, 7, 9],
        "conv_dilats": [1, 1, 1, 1],
        "LSK_kernels": [5, 7, 5, 5],
        "LSK_dilats": [1, 3, 1, 1],
        "LSK_mid_kernel": 7,
        "encoderDownKernels": [3, 5, 7,9],
        "fra_kernels": [3, 5, 7,9],
        "fra_dilates": [1, 1, 1,1],
        "MSAA_pool_kernel": 7,
        "MSAA_kernels": [3, 5, 7,9],
        "MSAA_dilats": [1, 1, 1,1],
        "decoderUpKernels": [3, 5, 7,9],
        "tdim": int(512*4),
        "tail_kernel": 5
    }

    condition_info = torch.randn(batch_size, 5, img_H, img_W).to(device)
    x_t = torch.randn(batch_size, 1, img_H, img_W).to(device)
    t = torch.rand(batch_size)
    
    # 创建网络实例
    model = velocity_UNet(net_info_dict)
    get_gpu_info.print_gpu_memory("Conv3x3_DownSample GPU info",x_t,model,temb = t,condition = condition_info)


if __name__ == '__main__':
    # test_TimeEmbedding()

    velocity_UNet_test()
    # ConditionalEmbedding_test()
    # attan_block_test()
    # LSK_test()
    # LSK_test()
    # ResConv_test()



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






        # conv_layer_num = len(self.encoder_conv_list)
   
        # time_encode_outputs = []
        # time_decode_outputs = []
        # x_in = torch.cat([x_t, condition_info], dim=1)

        # for i in range(len(self.Time_encode_Embeddings)):
        #     time_encode_outputs.append(self.Time_encode_Embeddings[i](t))
        #     time_decode_outputs.append(self.Time_decode_Embeddings[i](t))
        
        # # 编码器路径
        # encoder_outs = []
        # encoder_outs.append(x_in)
        # for i in range(len(self.encoder_conv_list)):
        #     x_in = self.encoder_down_list[i](self.encoder_conv_list[i](x_in,time_encode_outputs[conv_layer_num-i]))
        #     encoder_outs.append(x_in)

        # center_out = self.conv_center(encoder_outs[-1])
        # for i in range(len(conv_layer_num)):
            
        #     center_out = self.decode_up_list[i]( self.encoder_conv_list[i](torch.cat([center_out, self.mid_attn_list[conv_layer_num-i](center_out)], dim=1),time_decode_outputs[i]) )

        # return self.tail(torch.cat([center_out, self.mid_attn_list[conv_layer_num-i](encoder_outs[0])], dim=1))






# class velocity_UNet(nn.Module):
#     # 修改上卷积方法
#
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
#         C_down_list = [64, 128, 256, 512]
#     C_list_attn = torch.tensor([64, 64, 128, 128, 128])
#     attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
#     net = velocity_UNet(T,BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)
#     output = net(x_t, t, condition_info)
#     print(f"Output shape: {output.shape}")





