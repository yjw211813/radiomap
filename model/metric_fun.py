import torch
import torch.nn as nn

class NMSE(nn.Module):
    def __init__(self, eps=1e-9):
        """
        NMSE（归一化均方误差）损失函数
        eps: 防止分母为零的小值
        """
        super(NMSE, self).__init__()
        self.eps = eps

    def forward(self, outputs, targets):
        """
        前向传播计算损失
        参数:
            outputs: 网络输出，形状(batch_size, channels, H, W)
            targets: 目标值，形状(batch_size, channels, H, W)
        返回:
            NMSE损失值
        """
        # 计算分子：预测值与真实值的均方误差
        numerator = torch.sum((outputs - targets) ** 2, dim=[1, 2, 3])

        # 计算分母：真实值的平方和
        denominator = torch.sum(targets ** 2, dim=[1, 2, 3]) + self.eps

        # 计算每个样本的NMSE
        nmse_per_sample = numerator / denominator

        # 返回batch平均损失
        return torch.mean(nmse_per_sample)

def NMSE_test():
    # 创建损失函数对象
    nmse_loss = NMSE()

    # 模拟数据
    outputs = torch.randn(8, 1, 256, 256, requires_grad=True)  # 需要梯度
    targets = torch.randn(8, 1, 256, 256)

    # 计算损失
    loss = nmse_loss(outputs, targets)
    print(f"NMSE Loss: {loss.item()}")

    # 反向传播
    loss.backward()
    print("Gradients calculated successfully")


class SSIM(nn.Module):
    def __init__(self, L=1.0, k1=0.01, k2=0.03):
        """
        SSIM（结构相似性指数）损失函数

        参数:
            L: 像素值的动态范围（对于[0,1]范围的图像，L=1；对于[0,255]范围的图像，L=255）
            k1, k2: SSIM计算中的常数，通常设为0.01和0.03
        """
        super(SSIM, self).__init__()
        self.L = L
        self.k1 = k1
        self.k2 = k2

        # 计算常数C1, C2
        self.C1 = (k1 * L) ** 2
        self.C2 = (k2 * L) ** 2

    def forward(self, outputs, targets):
        """
        计算SSIM损失

        参数:
            outputs: 网络输出，形状(batch_size, channels, H, W)
            targets: 目标值，形状(batch_size, channels, H, W)

        返回:
            SSIM损失值（1 - SSIM）
        """
        batch_size, channels, H, W = outputs.size()

        # 计算均值
        mu_x = torch.mean(outputs, dim=[2, 3])  # (batch_size, channels)
        mu_y = torch.mean(targets, dim=[2, 3])

        # 计算方差
        var_x = torch.var(outputs, dim=[2, 3], unbiased=False)  # (batch_size, channels)
        var_y = torch.var(targets, dim=[2, 3], unbiased=False)

        # 计算协方差
        # 先将图像展平
        x_flat = outputs.view(batch_size, channels, -1)  # (batch_size, channels, H*W)
        y_flat = targets.view(batch_size, channels, -1)

        # 计算协方差
        cov_xy = torch.mean(x_flat * y_flat, dim=2) - mu_x * mu_y

        # 根据公式(30)计算SSIM
        # 分子部分: (2μ_x μ_y + C1) * (2σ_xy + C2)
        numerator = (2 * mu_x * mu_y + self.C1) * (2 * cov_xy + self.C2)

        # 分母部分: (μ_x² + μ_y² + C1) * (σ_x² + σ_y² + C2)
        denominator = (mu_x.pow(2) + mu_y.pow(2) + self.C1) * (var_x + var_y + self.C2)

        # 计算每个通道的SSIM
        ssim_per_channel = numerator / (denominator + 1e-6)  # 添加小量防止除零

        # 对通道和batch取平均
        ssim_val = torch.mean(ssim_per_channel)

        # 返回SSIM损失 (1 - SSIM)，因为SSIM越大表示越相似
        return 1 - ssim_val

def SSIM_test():
    # 创建损失函数对象
    ssim_loss = SSIM(L=1.0)  # 假设图像在[0,1]范围内

    # 模拟数据
    outputs = torch.rand(8, 1, 256, 256, requires_grad=True)  # 需要梯度
    targets = torch.rand(8, 1, 256, 256)

    # 计算损失
    loss = ssim_loss(outputs, targets)
    print(f"SSIM Loss: {loss.item():.6f}")

    # 反向传播测试
    loss.backward()
    print("Gradients calculated successfully")


class PSNR(nn.Module):
    def __init__(self, r=1.0, eps=1e-8):
        """
        PSNR（峰值信噪比）损失函数

        参数:
            r: 输入图像数据的最大可能值（对于[0,1]范围的图像，r=1；对于[0,255]范围的图像，r=255）
            eps: 防止数值不稳定的小量
        """
        super(PSNR, self).__init__()
        self.r = r
        self.eps = eps
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        """
        计算PSNR损失

        参数:
            outputs: 网络输出，形状(batch_size, channels, H, W)
            targets: 目标值，形状(batch_size, channels, H, W)

        返回:
            PSNR损失值（负的PSNR，因为PSNR越大越好，而损失应该越小）
        """
        # 计算每个样本的MSE
        mse_per_sample = self.mse_loss(outputs, targets)
        # 在空间维度上求平均 (H, W)
        mse_per_sample = torch.mean(mse_per_sample, dim=[1, 2, 3])

        # 计算每个样本的PSNR
        # PSNR = 10 * log10(r^2 / MSE)
        psnr_per_sample = 10 * torch.log10(self.r ** 2 / (mse_per_sample + self.eps))

        # 计算平均PSNR
        psnr_val = torch.mean(psnr_per_sample)

        # 返回负的PSNR作为损失（因为PSNR越大表示质量越好）
        return -psnr_val

def PSNR_test():
    # 创建损失函数对象，假设图像在[0,1]范围内
    psnr_loss = PSNR(r=1.0)

    # 模拟数据
    outputs = torch.rand(8, 1, 256, 256, requires_grad=True)
    targets = torch.rand(8, 1, 256, 256)

    # 计算损失
    loss = psnr_loss(outputs, targets)
    print(f"PSNR Loss: {loss.item():.6f}")

    # 反向传播测试
    loss.backward()
    print("Gradients calculated successfully")


# 使用示例
if __name__ == "__main__":
    # NMSE_test()
    SSIM_test()

