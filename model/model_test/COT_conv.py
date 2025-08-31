import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 显示初始显存情况
if torch.cuda.is_available():
    print(f"初始显存使用: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

# 定义参数
bs = 8  # 批量大小
c = 256  # 通道数
h = 64  # 高度
w = 64  # 宽度
kernel_size = 9  # 卷积核大小

# 创建随机输入
v = torch.randn(bs, c, h, w).to(device)  # 值张量
att = torch.randn(bs, c, kernel_size * kernel_size, h, w).to(device)  # 注意力权重（卷积核）

# 显示创建张量后的显存情况
if torch.cuda.is_available():
    print(f"创建输入张量后显存使用: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

# 方法1：循环实现
def loop_convolution(v, att, kernel_size):
    bs, c, h, w = v.shape
    padding = (kernel_size - 1) // 2
    v_padded = F.pad(v, (padding, padding, padding, padding), mode='constant', value=0)

    output_loop = torch.zeros_like(v).to(device)

    for i in range(h):
        for j in range(w):
            # 获取当前位置的邻域
            neighborhood = v_padded[:, :, i:i + kernel_size, j:j + kernel_size]

            # 获取当前位置对应的卷积核
            kernel = att[:, :, :, i, j].reshape(bs, c, kernel_size, kernel_size)

            # 执行逐通道卷积操作
            result = (neighborhood * kernel).sum(dim=(2, 3))

            # 将结果放入输出张量的对应位置
            output_loop[:, :, i, j] = result

    return output_loop


# 方法2：使用unfold的向量化实现
def unfold_convolution(v, att, kernel_size):
    bs, c, h, w = v.shape
    padding = (kernel_size - 1) // 2
    v_padded = F.pad(v, (padding, padding, padding, padding), mode='constant', value=0)

    # 使用unfold操作获取所有局部区域
    v_unfold = F.unfold(v_padded, kernel_size=kernel_size)  # 形状: (B, C*k*k, H*W)
    v_unfold = v_unfold.view(bs, c, kernel_size * kernel_size, h, w)  # 重塑为 (B, C, k*k, H, W)

    # 直接进行元素乘法和求和
    output_unfold = (v_unfold * att).sum(dim=2)  # 形状: (B, C, H, W)

    return output_unfold


if __name__ == '__main__':

    # 计算两种方法的结果
    output_loop = loop_convolution(v, att, kernel_size)
    if torch.cuda.is_available():
        print(f"循环实现后显存使用: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    output_unfold = unfold_convolution(v, att, kernel_size)
    if torch.cuda.is_available():
        print(f"Unfold实现后显存使用: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    # 比较两种方法的结果
    print("循环实现结果形状:", output_loop.shape)
    print("Unfold实现结果形状:", output_unfold.shape)
    print("两种实现的最大差异:", torch.max(torch.abs(output_loop - output_unfold)).item())
    print("两种实现的平均差异:", torch.mean(torch.abs(output_loop - output_unfold)).item())

    # 检查两种实现是否等价（考虑浮点数精度误差）
    if torch.allclose(output_loop, output_unfold, atol=1e-6):
        print("✓ 两种实现等价")
    else:
        print("✗ 两种实现不等价")

    # 性能比较
    import time

    # 测试循环实现的性能
    start_time = time.time()
    output_loop = loop_convolution(v, att, kernel_size)
    loop_time = time.time() - start_time

    # 测试unfold实现的性能
    start_time = time.time()

    output_unfold = unfold_convolution(v, att, kernel_size)
    unfold_time = time.time() - start_time

    print(f"\n性能比较:")
    print(f"循环实现时间: {loop_time:.4f} 秒 (10次运行)")
    print(f"Unfold实现时间: {unfold_time:.4f} 秒 (10次运行)")
    print(f"Unfold实现比循环实现快 {loop_time / unfold_time:.2f} 倍")

    # 显示最终显存情况
    if torch.cuda.is_available():
        print(f"最终显存使用: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        print(f"峰值显存使用: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")