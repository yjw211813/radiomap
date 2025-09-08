# -*- coding: utf-8 -*-
"""
REM-GAN模型训练脚本

优化内容：
1. 增加详细注释，提高代码可读性
2. 优化设备管理，使其更灵活
3. 改进梯度计算函数，使用更高效的实现
4. 重构训练循环，提高代码清晰度
5. 优化超参数管理
6. 改进损失计算和记录方式
7. 增强模型保存和恢复功能

@author: Achintha
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from model.rem_gan import modules, loss
from data.lib import REM_GAN_loaders
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from model.rem_gan.EncoderModels import ResnetGenerator, Discriminator
from tqdm import tqdm
from skimage.segmentation import slic
import scipy.stats as stats

# 超参数配置
TV_WEIGHT = 1e-7
BATCH_SIZE = 30
EPOCHS = 50
LEARNING_RATE = 0.001
LR_DECAY_EPOCHS = 25
DEVICE_ID = 'cuda:3' if torch.cuda.is_available() else 'cpu'


class GANTrainer:
    """REM-GAN模型训练器"""

    def __init__(self, netD, netG, trainset, valset=[], testset=[], phase='first',
                 batch_size=15, experiment_path='', device=DEVICE_ID):
        """
        初始化GAN训练器

        参数:
            netD: 判别器模型
            netG: 生成器模型
            trainset: 训练数据集
            valset: 验证数据集
            testset: 测试数据集
            phase: 训练阶段 ('first' 或其它)
            batch_size: 批处理大小
            experiment_path: 实验数据保存路径
            device: 训练设备
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.n_segments = 100
        self.phase = phase
        self.experiment_path = experiment_path

        # 创建数据加载器
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        # 初始化模型
        self.netG = netG.to(self.device)
        self.netD = netD.to(self.device)

        # 定义优化器
        self.optimG = optim.Adam(self.netG.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

        # 定义损失函数
        self.lossD = nn.BCEWithLogitsLoss()
        self.lossG = nn.MSELoss()
        self.lossGS = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.lossMsSSIM = loss.MS_SSIM_L1_LOSS()
        self.lossL1 = nn.L1Loss()

        # 训练历史记录
        self.loss_history = {
            'D': [], 'G': [], 'MSE': [],
            'D_avg': [], 'G_avg': [], 'MSE_avg': [],
            'val_loss': []
        }

        # 创建实验目录
        os.makedirs(experiment_path, exist_ok=True)

    def tv_loss(self, y_hat):
        """计算总变差损失"""
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        return TV_WEIGHT * (diff_i + diff_j)

    def compute_gradients(self, img):
        """计算批次图像的梯度"""
        # 确保输入是4D张量
        assert img.dim() == 4, "Input must be a 4D tensor [batch, channel, height, width]"

        # 定义Sobel滤波器
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_xy = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
                                dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_yx = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

        # 卷积操作，每个滤波器的输出是 [batch_size, 1, height, width]
        G_x = F.conv2d(img, sobel_x, padding=1)
        G_y = F.conv2d(img, sobel_y, padding=1)
        G_xy = F.conv2d(img, sobel_xy, padding=1)
        G_yx = F.conv2d(img, sobel_yx, padding=1)

        # 拼接成 [batch_size, 4, height, width]
        return torch.cat([G_x, G_y, G_xy, G_yx], dim=1)

    def power_spectrum_loss(self, images, E=100):
        """计算功率谱损失"""
        image = images.detach().cpu().numpy().astype(int)
        npix = images.shape[1]

        # 计算傅里叶变换和功率谱
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2

        # 计算频率
        kfreq = np.fft.fftfreq(npix) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

        # 扁平化处理
        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.reshape(images.shape[0], -1)

        # 计算分箱统计
        kbins = np.arange(0.5, npix // 2 + 1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])

        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                             statistic="mean",
                                             bins=kbins)
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

        # 获取前E个最重要的频率分量
        ind = np.argpartition(Abins, E)
        return torch.FloatTensor(ind).to(self.device)

    def one_hot_encode(self, labels, n_classes):
        """将标签转换为one-hot编码"""
        return F.one_hot(labels, num_classes=n_classes)

    def concatenate_vectors(self, x, y):
        """连接两个张量"""
        return torch.cat((x.float(), y.float()), 1)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.netG.train()
        self.netD.train()

        epoch_d_loss = []
        epoch_g_loss = []
        epoch_mse_loss = []

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for batch_idx, (inps, gts) in enumerate(progress_bar):
            inps, gts = inps.to(self.device), gts.to(self.device)
            up_sampled = inps[:, 3, :, :].unsqueeze(1)
            inps = inps[:, :3, :, :]

            ############################
            # 训练判别器
            ############################
            self.optimD.zero_grad()

            # 创建真实和假的标签
            real_labels = torch.ones(inps.size(0), dtype=torch.long, device=self.device)
            fake_labels = torch.zeros(inps.size(0), dtype=torch.long, device=self.device)

            # 生成one-hot编码
            one_hot_real = self.one_hot_encode(real_labels, 2)[:, :, None, None]
            one_hot_fake = self.one_hot_encode(fake_labels, 2)[:, :, None, None]

            # 扩展维度以匹配图像大小
            one_hot_real = one_hot_real.repeat(1, 1, inps.shape[2], inps.shape[3])
            one_hot_fake = one_hot_fake.repeat(1, 1, inps.shape[2], inps.shape[3])

            # 生成假图像
            with torch.no_grad():
                fake_images, _ = self.netG(inps)

            # 拼接图像和标签
            fake_input = self.concatenate_vectors(fake_images, one_hot_fake)
            real_input = self.concatenate_vectors(gts, one_hot_real)

            # 判别器前向传播
            pred_fake = self.netD(fake_input.detach())
            pred_real = self.netD(real_input)

            # 计算判别器损失
            loss_fake = self.lossD(pred_fake, torch.zeros_like(pred_fake))
            loss_real = self.lossD(pred_real, torch.ones_like(pred_real))
            d_loss = (loss_fake + loss_real) / 2

            # 反向传播和优化
            d_loss.backward()
            self.optimD.step()

            ############################
            # 训练生成器
            ############################
            self.optimG.zero_grad()

            # 生成假图像
            fake_images, _ = self.netG(inps)
            fake_input = self.concatenate_vectors(fake_images, one_hot_real)

            # 判别器前向传播
            pred_fake = self.netD(fake_input)

            # 计算生成器损失
            if self.phase == 'first':
                g_adv_loss = self.lossD(pred_fake, torch.ones_like(pred_fake))
                g_mse_loss = self.lossG(fake_images, gts)

                # 计算梯度损失
                grad_fake = self.compute_gradients(fake_images)
                grad_up = self.compute_gradients(up_sampled)
                grad_sim_loss = 1 - torch.mean(self.lossGS(grad_up, grad_fake))

                # 计算TV损失
                tv_loss = self.tv_loss(fake_images)

                # 总损失
                g_loss = 10 * g_adv_loss + g_mse_loss + 10 * tv_loss + 10 * grad_sim_loss
            else:
                g_adv_loss = self.lossD(pred_fake, torch.ones_like(pred_fake))
                ssim_loss = self.lossMsSSIM(fake_images, gts)
                l1_loss = self.lossG(fake_images, gts)

                # 计算功率谱损失
                power_loss_val = self.lossL1(
                    self.power_spectrum_loss(up_sampled.squeeze(1)),
                    self.power_spectrum_loss(fake_images.squeeze(1))
                )

                # 计算TV损失
                tv_loss = self.tv_loss(fake_images)

                # 计算超像素一致性损失
                seg_loss = self._compute_superpixel_loss(inps[:, 2, :, :], fake_images.squeeze(1))

                # 总损失
                g_loss = (g_adv_loss + 100 * l1_loss + 0.001 * tv_loss +
                          84 * ssim_loss + power_loss_val + seg_loss)

            # 反向传播和优化
            g_loss.backward()
            self.optimG.step()

            # 计算MSE损失
            with torch.no_grad():
                mse_loss = self.lossG(fake_images, gts)

            # 记录损失
            epoch_d_loss.append(d_loss.item())
            epoch_g_loss.append(g_loss.item())
            epoch_mse_loss.append(mse_loss.item())

            # 更新进度条
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}'
            })

        return epoch_d_loss, epoch_g_loss, epoch_mse_loss

    def _compute_superpixel_loss(self, input_imgs, fake_imgs):
        """计算超像素一致性损失"""
        batch_size = input_imgs.size(0)
        segments_i = torch.zeros(batch_size, self.n_segments, device=self.device)
        segments_f = torch.zeros(batch_size, self.n_segments, device=self.device)

        for i in range(batch_size):
            # 将数据移至CPU进行超像素分割
            img_np = input_imgs[i].detach().cpu().numpy()
            superpixels = slic(img_np, n_segments=self.n_segments, sigma=5)

            for n, seg_id in enumerate(np.unique(superpixels)):
                if n >= self.n_segments:
                    break

                # 找到超像素中的最大响应点
                x, y = np.where(superpixels == seg_id)
                max_idx = np.argmax(img_np[x, y])
                x_max, y_max = x[max_idx], y[max_idx]

                # 记录输入和生成图像在最大响应点的值
                segments_i[i, n] = input_imgs[i, x_max, y_max]
                segments_f[i, n] = fake_imgs[i, x_max, y_max]

        return self.lossL1(segments_i, segments_f)

    def validate(self):
        """在验证集上评估模型"""
        self.netG.eval()
        self.netD.eval()

        val_losses = []

        with torch.no_grad():
            for inps, gts in self.val_loader:
                inps, gts = inps.to(self.device), gts.to(self.device)
                up_sampled = inps[:, 3, :, :].unsqueeze(1)
                inps = inps[:, :3, :, :]

                fake_images, _ = self.netG(inps)
                loss = self.lossG(fake_images, gts)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def train(self, epochs=EPOCHS):
        """训练模型"""
        best_loss = float('inf')
        best_model_wts = {
            'G': copy.deepcopy(self.netG.state_dict()),
            'D': copy.deepcopy(self.netD.state_dict())
        }

        # 学习率调度器
        lr_scheduler_G = lr_scheduler.StepLR(self.optimG, step_size=LR_DECAY_EPOCHS, gamma=0.1)
        lr_scheduler_D = lr_scheduler.StepLR(self.optimD, step_size=LR_DECAY_EPOCHS, gamma=0.1)

        for epoch in range(epochs):
            # 训练一个epoch
            d_losses, g_losses, mse_losses = self.train_epoch(epoch)

            # 计算平均损失
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)
            avg_mse_loss = np.mean(mse_losses)

            # 验证
            val_loss = self.validate()

            # 更新学习率
            lr_scheduler_G.step()
            lr_scheduler_D.step()

            # 记录损失历史
            self.loss_history['D'].extend(d_losses)
            self.loss_history['G'].extend(g_losses)
            self.loss_history['MSE'].extend(mse_losses)
            self.loss_history['D_avg'].append(avg_d_loss)
            self.loss_history['G_avg'].append(avg_g_loss)
            self.loss_history['MSE_avg'].append(avg_mse_loss)
            self.loss_history['val_loss'].append(val_loss)

            # 打印统计信息
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(
                f'  D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, MSE: {avg_mse_loss:.4f}, Val_MSE: {val_loss:.4f}')
            print(f'  LR_G: {lr_scheduler_G.get_last_lr()[0]:.6f}, LR_D: {lr_scheduler_D.get_last_lr()[0]:.6f}')

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = {
                    'G': copy.deepcopy(self.netG.state_dict()),
                    'D': copy.deepcopy(self.netD.state_dict())
                }

                # 保存模型
                torch.save(self.netG.state_dict(),
                           os.path.join(self.experiment_path, 'best_generator.pth'))
                torch.save(self.netD.state_dict(),
                           os.path.join(self.experiment_path, 'best_discriminator.pth'))

                print(f'  Saved best model with validation loss: {val_loss:.4f}')

        # 保存损失历史
        self._save_loss_history()

        return best_model_wts

    def _save_loss_history(self):
        """保存损失历史到CSV文件"""
        # 保存每个batch的损失
        pd.DataFrame({
            'D_loss': self.loss_history['D'],
            'G_loss': self.loss_history['G'],
            'MSE': self.loss_history['MSE']
        }).to_csv(os.path.join(self.experiment_path, 'loss_per_batch.csv'), index=False)

        # 保存每个epoch的平均损失
        pd.DataFrame({
            'D_loss_avg': self.loss_history['D_avg'],
            'G_loss_avg': self.loss_history['G_avg'],
            'MSE_avg': self.loss_history['MSE_avg'],
            'val_loss': self.loss_history['val_loss']
        }).to_csv(os.path.join(self.experiment_path, 'loss_per_epoch.csv'), index=False)


def setup_datasets(setup_type=1):
    """
    根据设置类型准备数据集

    参数:
        setup_type: 数据集设置类型 (1: 均匀1%, 2: 双侧1%和10%, 3: 非均匀1%到10%)

    返回:
        train_set, val_set, test_set: 训练、验证和测试数据集
    """
    setups = ['uniform', 'twoside', 'nonuniform']
    setup_name = setups[setup_type - 1]

    if setup_type == 1:
        # 均匀1%采样
        train_set = REM_GAN_loaders.RadioUNet_s(phase="train", fix_samples=655,
                                                num_samples_low=10, num_samples_high=300)
        val_set = REM_GAN_loaders.RadioUNet_s(phase="val", fix_samples=655,
                                              num_samples_low=10, num_samples_high=300)
        test_set = REM_GAN_loaders.RadioUNet_s(phase="test", fix_samples=655,
                                               num_samples_low=10, num_samples_high=300)

    elif setup_type == 2:
        # 双侧1%和10%采样
        train_set = REM_GAN_loaders.RadioUNet_s(phase="train", fix_samples=1,
                                                num_samples_low=655, num_samples_high=655 * 10)
        val_set = REM_GAN_loaders.RadioUNet_s(phase="val", fix_samples=1,
                                              num_samples_low=655, num_samples_high=655 * 10)
        test_set = REM_GAN_loaders.RadioUNet_s(phase="test", fix_samples=1,
                                               num_samples_low=655, num_samples_high=655 * 10)

    else:
        # 非均匀1%到10%采样
        train_set = REM_GAN_loaders.RadioUNet_s(phase="train", fix_samples=0,
                                                num_samples_low=655, num_samples_high=655 * 10)
        val_set = REM_GAN_loaders.RadioUNet_s(phase="val", fix_samples=0,
                                              num_samples_low=655, num_samples_high=655 * 10)
        test_set = REM_GAN_loaders.RadioUNet_s(phase="test", fix_samples=0,
                                               num_samples_low=655, num_samples_high=655 * 10)

    return train_set, val_set, test_set, setup_name


if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 配置实验参数
    SETUP_TYPE = 1  # 1: 均匀1%, 2: 双侧1%和10%, 3: 非均匀1%到10%
    EXP_INDEX = 1  # 实验索引

    # 准备数据集
    train_set, val_set, test_set, setup_name = setup_datasets(SETUP_TYPE)

    # 设置设备
    device = torch.device(DEVICE_ID)
    print(f"Using device: {device}")

    # 创建实验目录
    exp_path = f"{setup_name}_{EXP_INDEX}"
    os.makedirs(exp_path, exist_ok=True)

    # 初始化模型
    netG = modules.RadioWNet(phase="firstU").to(device)
    netD = Discriminator(device).to(device)

    # 创建训练器并开始训练
    trainer = GANTrainer(netD, netG, train_set, val_set, test_set,
                         phase='first', batch_size=BATCH_SIZE,
                         experiment_path=exp_path, device=device)

    best_models = trainer.train(epochs=EPOCHS)

    # 保存最终模型
    torch.save(best_models['G'], os.path.join(exp_path, 'final_generator.pth'))
    torch.save(best_models['D'], os.path.join(exp_path, 'final_discriminator.pth'))

    print("Training completed!")