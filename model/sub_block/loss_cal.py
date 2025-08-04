import torch
import torch.nn as nn
import random
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from model.sub_block.metric_fun import SSIMLoss
import torch.nn.functional as F




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

def DynamicLoss_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    targets = torch.randn(32, 1, 256, 256)
    outputs = torch.randn(32, 1, 256, 256)
    dynamic_loss = DynamicLoss(
        mse_weight=1.0,
        ssim_weight=0.005,
        switch_epoch=10,
        mix_prob=0.2  # 20%概率使用混合损失
    ).to(device)

    loss = dynamic_loss(outputs, targets, 20)
    print("Output shape:", loss.item())


class FourierLoss(nn.Module):
    def __init__(self,
                 mse_weight: float = 1.0,
                 fourier_amp_weight: float = 0.4,
                 fourier_phase_weight: float = 0.1,
                 switch_epoch: int = 10,
                 mix_prob: float = 0.2,
                 log_amp: bool = True,  # 新增：是否对振幅应用对数变换
                 eps: float = 1e-8):
        """
        动态混合损失函数，结合像素级MSE损失和傅里叶域损失

        参数:
            mse_weight (float): MSE损失的权重因子
            fourier_amp_weight (float): 傅里叶振幅损失的权重因子
            fourier_phase_weight (float): 傅里叶相位损失的权重因子
            switch_epoch (int): 开始使用傅里叶损失的训练轮次
            mix_prob (float): 使用混合损失的概率 [0.0, 1.0]
            log_amp (bool): 是否对振幅应用对数变换（推荐True）
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
        self.log_amp = log_amp
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

        # 提取振幅分量
        outputs_amp = torch.abs(outputs_fft)
        targets_amp = torch.abs(targets_fft)

        # 对振幅应用对数变换（推荐）
        if self.log_amp:
            # 使用 log(1 + amp) 避免 log(0) 问题
            outputs_amp = torch.log1p(outputs_amp)  # 等同于 log(1 + amp)
            targets_amp = torch.log1p(targets_amp)

        # 使用角度函数提取相位（弧度制）
        outputs_phase = torch.angle(outputs_fft)
        targets_phase = torch.angle(targets_fft)

        # 计算振幅损失（MSE）
        amp_loss = self.mse(outputs_amp, targets_amp)

        # 计算相位损失（MSE）
        # 相位值在[-π, π]之间，不需要对数变换
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


def FourierLoss_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    targets = torch.randn(32, 1, 256, 256)
    outputs = torch.randn(32, 1, 256, 256)
    fourierLoss = FourierLoss(
        mse_weight=1.0,
        fourier_amp_weight=0.4,
        fourier_phase_weight=0.1,
        switch_epoch=10,
        mix_prob=0.2
    ).to(device)

    loss = fourierLoss(outputs, targets, 20)
    print("Output shape:", loss.item())




class MaskFourierLoss(nn.Module):
    def __init__(self,
                 mse_weight: float = 1.0,
                 fourier_phase_weight: float = 0.1,
                 low_freq_radius: float = 0.2,  # 低频区域半径比例
                 low_freq_amp_weight: float = 0.05,  # 低频区域振幅权重
                 high_freq_amp_weight: float = 0.35,  # 高频区域振幅权重
                 switch_epoch: int = 10,
                 mix_prob: float = 0.2,
                 log_amp: bool = True,
                 eps: float = 1e-8):
        """
        动态混合损失函数，结合像素级MSE损失和傅里叶域损失
        参数:
            mse_weight (float): MSE损失的权重因子
            fourier_phase_weight (float): 傅里叶相位损失的权重因子
            low_freq_radius (float): 低频区域半径比例 (0-1)
            low_freq_amp_weight (float): 低频区域振幅损失权重
            high_freq_amp_weight (float): 高频区域振幅损失权重
            switch_epoch (int): 开始使用傅里叶损失的训练轮次
            mix_prob (float): 使用混合损失的概率 [0.0, 1.0]
            log_amp (bool): 是否对振幅应用对数变换（推荐True）
            eps (float): 数值稳定性参数，防止除零错误
        """
        super(MaskFourierLoss, self).__init__()
        self.mse = nn.MSELoss()

        # 权重参数
        self.mse_weight = mse_weight
        self.fourier_phase_weight = fourier_phase_weight
        self.low_freq_radius = low_freq_radius
        self.low_freq_amp_weight = low_freq_amp_weight
        self.high_freq_amp_weight = high_freq_amp_weight

        # 训练控制参数
        self.switch_epoch = switch_epoch
        self.mix_prob = mix_prob
        self.log_amp = log_amp
        self.eps = eps
        # 验证参数有效性
        self._validate_parameters()

    def _validate_parameters(self):
        """验证输入参数的有效性"""
        assert 0 <= self.mix_prob <= 1.0, "mix_prob 必须在 [0, 1] 范围内"
        assert self.switch_epoch >= 0, "switch_epoch 必须是非负整数"
        assert self.mse_weight >= 0, "mse_weight 必须是非负数"
        assert 0 <= self.low_freq_radius <= 1, "low_freq_radius 必须在 [0, 1] 范围内"
        assert self.low_freq_amp_weight >= 0, "low_freq_amp_weight 必须是非负数"
        assert self.high_freq_amp_weight >= 0, "high_freq_amp_weight 必须是非负数"
        assert self.fourier_phase_weight >= 0, "fourier_phase_weight 必须是非负数"

    def _create_frequency_mask(self, height: int, width: int) -> torch.Tensor:
        """
        创建频率区域掩码，区分低频和高频区域
        参数:
            height: 图像高度
            width: 图像宽度
        返回:
            掩码张量，低频区域为True，高频区域为False
        """
        # 计算中心点
        center_y = height // 2
        center_x = width // 2

        # 计算最大半径（取高度和宽度中较小值的一半）
        max_radius = min(height, width) * self.low_freq_radius / 2

        # 创建坐标网格
        y = torch.arange(height, dtype=torch.float32).view(-1, 1)
        x = torch.arange(width, dtype=torch.float32).view(1, -1)

        # 计算每个点到中心的距离
        dist_from_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        # 创建掩码（中心低频区域为True，外围高频区域为False）
        mask = dist_from_center <= max_radius

        return mask

    def _compute_fourier_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算傅里叶域损失（振幅 + 相位）

        参数:
            outputs: 模型输出张量 (B, C, H, W)
            targets: 目标张量 (B, C, H, W)

        返回:
            傅里叶损失值
        """
        # 获取图像尺寸
        B, C, H, W = outputs.shape

        # 计算2D FFT（最后两个维度）
        outputs_fft = torch.fft.fft2(outputs, dim=(-2, -1))
        targets_fft = torch.fft.fft2(targets, dim=(-2, -1))

        # 对FFT结果进行移位，使低频位于中心
        outputs_fft_shift = torch.fft.fftshift(outputs_fft, dim=(-2, -1))
        targets_fft_shift = torch.fft.fftshift(targets_fft, dim=(-2, -1))

        # 提取振幅分量
        outputs_amp = torch.abs(outputs_fft_shift)
        targets_amp = torch.abs(targets_fft_shift)

        # 对振幅应用对数变换（推荐）
        if self.log_amp:
            outputs_amp = torch.log1p(outputs_amp)  # 等同于 log(1 + amp)
            targets_amp = torch.log1p(targets_amp)

        # 使用角度函数提取相位（弧度制）
        outputs_phase = torch.angle(outputs_fft_shift)
        targets_phase = torch.angle(targets_fft_shift)

        # 创建频率区域掩码
        freq_mask = self._create_frequency_mask(H, W).to(outputs.device)

        # 分离低频和高频区域
        outputs_amp_low = outputs_amp[:, :, freq_mask]
        targets_amp_low = targets_amp[:, :, freq_mask]

        outputs_amp_high = outputs_amp[:, :, ~freq_mask]
        targets_amp_high = targets_amp[:, :, ~freq_mask]

        # 计算低频区域振幅损失
        if outputs_amp_low.numel() > 0:
            amp_loss_low = F.mse_loss(outputs_amp_low, targets_amp_low)
        else:
            amp_loss_low = torch.tensor(0.0, device=outputs.device)

        # 计算高频区域振幅损失
        if outputs_amp_high.numel() > 0:
            amp_loss_high = F.mse_loss(outputs_amp_high, targets_amp_high)
        else:
            amp_loss_high = torch.tensor(0.0, device=outputs.device)

        # 加权组合振幅损失
        amp_loss = (self.low_freq_amp_weight * amp_loss_low +
                    self.high_freq_amp_weight * amp_loss_high)

        # 计算相位损失（不分区域）
        phase_loss = F.mse_loss(outputs_phase, targets_phase)

        # 加权组合傅里叶损失
        fourier_loss = amp_loss + self.fourier_phase_weight * phase_loss

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

def MaskFourierLoss_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    targets = torch.randn(32, 1, 256, 256)
    outputs = torch.randn(32, 1, 256, 256)
    maskFourierLoss = MaskFourierLoss(
        mse_weight = 1.0,
        fourier_phase_weight = 0.1,
        low_freq_radius = 0.2,  # 低频区域半径比例
        low_freq_amp_weight = 0.05,  # 低频区域振幅权重
        high_freq_amp_weight = 0.35,  # 高频区域振幅权重
        switch_epoch =5,
        mix_prob= 0.2,
    ).to(device)

    loss = maskFourierLoss(outputs, targets, 20)
    print("Output shape:", loss.item())

if __name__ == '__main__':
    DynamicLoss_test()
    FourierLoss_test()
    MaskFourierLoss_test()