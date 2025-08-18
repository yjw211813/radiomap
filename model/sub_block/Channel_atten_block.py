import torch
import torch.nn as nn

'''
    常用的通道注意力模块
'''
# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Channel_attan2D(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Channel_attan2D, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.GELU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).reshape(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).reshape(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

def SE_Channel_attan2D_test():
    # 测试 SE_Block
    x = torch.randn(2, 32, 128, 128)  # 假设输入是一个 batch_size 为 2，通道数为 32，8x8 的特征图
    se_block = SE_Channel_attan2D(inchannel=32)
    output = se_block(x)
    print(output.shape)  # 应输出 (2, 32, 128, 128)


# 三入 相加 或者两入相加 得到组合特征

class SK_Channel_atten2D(nn.Module):
    def __init__(self, img_size,C_in,C_out,kernel_sizes,dilated_list):
        super(SK_Channel_atten2D, self).__init__()
        
        self.C_out = C_out
        self.num_branches = len(kernel_sizes)
        ratio = 8
        down_sample_size = 8

        # 定义分支 feature extracture spit
        drop_out = 0.05
        self.branches = nn.ModuleList()
        for i in range(self.num_branches):
            
            dilated_kernel_size = (kernel_sizes[i] - 1) * dilated_list[i] + 1
            padding_width = (dilated_kernel_size - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size= kernel_sizes[i],
                                stride=(1, 1),
                                dilation=dilated_list[i],
                                padding= padding_width),
                nn.BatchNorm2d(C_out),
                nn.Dropout(drop_out),
                nn.GELU())
            )
     
        # 分支融合后压缩模块
        # 使用迭代计算替代 log2 计算降采样次数
        current_size = img_size
        self.compress_info = nn.ModuleList()

        # 添加分层卷积降维直到达到目标尺寸
        while current_size > down_sample_size:
            self.compress_info.append(
                nn.Sequential(
                    nn.Conv2d(C_out, C_out, kernel_size=3, stride=2, groups=C_out, padding=1),
                    nn.GELU()
                )
            )
            current_size = current_size // 2  # 每次降采样尺寸减半

        self.compress_info.append(nn.Sequential(
            nn.Conv2d(C_out, C_out, kernel_size=8,groups=C_out),
            nn.GELU()))

        self.feature_transform = nn.Sequential(
            nn.Linear(C_out, C_out // ratio, bias=False),
            nn.BatchNorm1d(C_out // ratio),
            nn.GELU(),
            nn.Linear(C_out // ratio, self.num_branches * C_out, bias=False),
        )
        self.weight_layer = nn.Sigmoid()

    def forward(self, x):
        
        # 1. 通过各个分支处理输入
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))

        # 2. 将分支输出相加得到融合特征
        fused = torch.stack(branch_outputs, dim=0).sum(dim=0)


        # 3. 通过压缩模块降维
        for compress_block in self.compress_info:
            fused = compress_block(fused)
        # compressed = self.compress_info(fused)
        compressed = fused.reshape(x.size(0), self.C_out)  # 展平为(batch_size, C_out)
       
       # 4. 通过全连接层生成权重
        weights = self.feature_transform(compressed)
        # 5. 拆分成多个分支的权重
        weights_split = weights.view(x.size(0), self.num_branches, self.C_out)
        weights_split = self.sigmoid(weights_split)
        # 形状变为(b, num_branches, C_out, 1, 1)
        weights_split = weights_split.unsqueeze(-1).unsqueeze(-1)

        for i in range(self.num_branches):
            if i == 0:
                # 应用对应分支的权重
                final_output = branch_outputs[i] * weights_split[:, i, :, :, :]
            else:
                final_output += branch_outputs[i] * weights_split[:, i, :, :, :]
        return final_output

def SK_Channel_atten2D_test():
    # 测试参数
    img_size = 128
    C_in = 32
    C_out = 64
    kernel_sizes = [3, 5, 7]
    dilated_list = [1, 2, 3]
    
    # 创建测试输入和模型
    x = torch.randn(2, C_in, img_size, img_size)
    model = SK_Channel_atten2D(img_size, C_in, C_out, kernel_sizes, dilated_list)
    
    # 前向传播
    output = model(x)
    print("Output shape:", output.shape)  # 应输出 (2, 64, 128, 128)


# 运行测试
if __name__ == "__main__":
    SK_Channel_atten2D_test()

# compress_layers = []
# # 动态构建降采样层
# while current_size > down_sample_size:
#     compress_layers.extend([
#         nn.Conv2d(C_out, C_out, kernel_size=3,
#                   stride=2, groups=C_out, padding=1),
#         nn.GELU()
#     ])
#     current_size = current_size // 2
# # 添加最后的自适应层
# compress_layers.extend([
#     nn.Conv2d(C_out, C_out, kernel_size=current_size, groups=C_out),
#     nn.GELU()
# ])
# self.compress_info = nn.Sequential(*compress_layers)

