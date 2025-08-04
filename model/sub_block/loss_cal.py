import torch
import torch.nn as nn
import random
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from model.sub_block.metric_fun import SSIMLoss

class DynamicLoss(nn.Module):
    def __init__(self,
                 mse_weight=1.0,
                 ssim_weight=0.01,
                 switch_epoch=10,
                 mix_prob=1.0,
                 data_range=1.0):
        """
        动态混合损失函数
        参数:
            mse_weight: MSE损失权重
            ssim_weight: SSIM损失权重
            switch_epoch: 开始混合损失的epoch
            mix_prob: 在switch_epoch后使用混合损失的概率
            data_range: SSIM的数据范围
        """
        super(DynamicLoss,self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(data_range=data_range)
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.switch_epoch = switch_epoch
        self.mix_prob = mix_prob

    def forward(self, outputs, targets, current_epoch):
        # 基础MSE损失
        mse_loss = self.mse(outputs, targets) * self.mse_weight

        # 10个epoch前只用MSE
        if current_epoch <= self.switch_epoch:
            return mse_loss

        # 10个epoch后按概率混合损失
        if random.random() < self.mix_prob:
            ssim_loss = self.ssim(outputs, targets) * self.ssim_weight
            return mse_loss + ssim_loss
        return mse_loss





class FourierLoss(nn.Module):
    def __init__(self,
                 mse_weight: float = 1.0,
                 fourier_amp_weight: float = 0.5,
                 fourier_phase_weight: float = 0.3,
                 switch_epoch: int = 10,
                 mix_prob: float = 1.0,
                 eps: float = 1e-8):
        """
        动态混合损失函数，结合像素级MSE损失和傅里叶域损失

        参数:
            mse_weight (float): MSE损失的权重因子
            fourier_amp_weight (float): 傅里叶振幅损失的权重因子
            fourier_phase_weight (float): 傅里叶相位损失的权重因子
            switch_epoch (int): 开始使用傅里叶损失的训练轮次
            mix_prob (float): 使用混合损失的概率 [0.0, 1.0]
            eps (float): 数值稳定性参数，防止log(0)
        """
        super(FourierLoss, self).__init__()
        self.mse = nn.MSELoss()

        # 权重参数
        self.mse_weight = mse_weight
        self.fourier_amp_weight = fourier_amp_weight
        self.fourier_phase_weight = fourier_phase_weight

        # 训练控制参数
        self.switch_epoch = switch_epoch
        self.mix_prob = mix_prob
        self.eps = eps  # 数值稳定性参数

        # 验证参数有效性
        self._validate_parameters()

    def _validate_parameters(self):
        """验证输入参数的有效性"""
        assert 0 <= self.mix_prob <= 1.0, "mix_prob 必须在 [0, 1] 范围内"
        assert self.switch_epoch >= 0, "switch_epoch 必须是非负整数"
        assert self.mse_weight >= 0, "mse_weight 必须是非负数"
        assert self.fourier_amp_weight >= 0, "fourier_amp_weight 必须是非负数"
        assert self.fourier_phase_weight >= 0, "fourier_phase_weight 必须是非负数"

    def _compute_fourier_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算傅里叶域损失（振幅 + 相位）
        参数:
            outputs: 模型输出张量 (B, C, H, W)
            targets: 目标张量 (B, C, H, W)
        返回:
            傅里叶损失值
        """
        # 计算2D FFT（最后两个维度）
        outputs_fft = torch.fft.fft2(outputs, dim=(-2, -1))
        targets_fft = torch.fft.fft2(targets, dim=(-2, -1))

        # 提取振幅和相位分量
        outputs_amp = torch.abs(outputs_fft)
        targets_amp = torch.abs(targets_fft)

        # 使用角度函数提取相位（弧度制）
        outputs_phase = torch.angle(outputs_fft)
        targets_phase = torch.angle(targets_fft)

        # 计算振幅损失（可选：使用对数振幅增强稳定性）
        amp_loss = self.mse(outputs_amp, targets_amp)

        # 计算相位损失（确保相位在有效范围内）
        phase_loss = self.mse(outputs_phase, targets_phase)

        # 加权组合傅里叶损失
        fourier_loss = (self.fourier_amp_weight * amp_loss +
                        self.fourier_phase_weight * phase_loss)

        return fourier_loss

    def forward(self,
                outputs: torch.Tensor,
                targets: torch.Tensor,
                current_epoch: int) -> torch.Tensor:
        """
        损失函数前向计算
        参数:
            outputs: 模型输出张量 (B, C, H, W)
            targets: 目标张量 (B, C, H, W)
            current_epoch: 当前训练轮次
        返回:
            组合损失值
        """
        # 基础像素级MSE损失
        mse_loss = self.mse(outputs, targets) * self.mse_weight

        # 在切换轮次前仅使用MSE损失
        if current_epoch < self.switch_epoch:
            return mse_loss

        # 在切换轮次后，按概率混合损失
        if random.random() < self.mix_prob:
            # 计算傅里叶域损失
            fourier_loss = self._compute_fourier_loss(outputs, targets)

            # 组合总损失
            total_loss = mse_loss + fourier_loss
            return total_loss

        # 不使用傅里叶损失的情况
        return mse_loss
