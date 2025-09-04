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



# resconv 参数部分 卷积 conv_kernels，conv_dilats  注意力 LSK_kernels，LSK_dilats，LSK_mid_kernel
# 下采样 多尺度 convDownKernels 上采样 多尺度 convUpKernels
# 应该有的通道主干 C_list =[64, 128, 256, 512]
# input_shape = [B, C, H, W] output_shape = [B, 1, H, W]

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

    condition_info = torch.randn(batch_size, 5, img_H, img_W)
    x_t = torch.randn(batch_size, 1, img_H, img_W)
    t = torch.rand(batch_size)
    
    # 创建网络实例
    model = velocity_UNet(net_info_dict)
    get_gpu_info.print_gpu_memory("Conv3x3_DownSample GPU info",x_t,model,temb = t,condition = condition_info)


if __name__ == '__main__':


    velocity_UNet_test()





