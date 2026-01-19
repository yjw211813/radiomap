# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:59:34 2022

trying loss2 > loss 1 : done

@author: Achintha

This code uses 1% from oneside 10% from the other side
use loadersUNETCGAN_f woth fix_sample = 1
"""

from __future__ import print_function, division
import os
import torch

import numpy as np
import math
import torch.optim as optim

from torch.optim import lr_scheduler
import torchvision
import shutil
from model.rem_gan import modules, loss
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.segmentation import slic
import scipy.stats as stats
import matplotlib.pyplot as plt
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim


TV_WEIGHT = 1e-7


def tvloss(y_hat):
    diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
    tv_loss = TV_WEIGHT * (diff_i + diff_j)
    return tv_loss


def gradient_img(img, device):
    img = img.squeeze(0)
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device))
    G_x = conv1(img)
    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).to(device))
    G_y = conv2(img)

    c = np.array([[-2, -1, -0], [-1, 0, 1], [0, 1, 2]])
    conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv3.weight = nn.Parameter(torch.from_numpy(c).float().unsqueeze(0).unsqueeze(0).to(device))
    G_xy = conv3(img)  # conv1(Variable(x)).data.view(1,x.shape[2],x.shape[3])

    d = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    conv4 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv4.weight = nn.Parameter(torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device))
    G_yx = conv4(img)  # (Variable(x)).data.view(1,x.shape[2],x.shape[3])

    G = torch.cat([G_x, G_y, G_xy, G_yx], dim=1)
    # G = torch.cat([G_x,G_y,G_xy,G_yx],dim=1)

    # G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return G


def power_loss(images, device, E=100):
    # print('images: ',images.shape)
    image = images.detach().cpu().numpy().astype(int)
    npix = images.shape[1]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image) ** 2

    kfreq = np.fft.fftfreq(npix) * npix

    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.reshape(images.shape[0], -1)
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    ind = np.argpartition(Abins, E)
    return torch.FloatTensor(ind).to(device)


def calc_loss_dense(pred, target, metrics):
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def calc_loss_sparse(pred, target, samples, metrics, num_samples):
    criterion = nn.MSELoss()
    loss = criterion(samples * pred, samples * target) * (256 ** 2) / num_samples
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs1)))


def concat_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1)
    return combined


def ohe_vector_from_labels(labels, n_classes):
    return F.one_hot(labels, num_classes=n_classes)


class REM_GAN_app():
    def __init__(self,  model_app_dict, dataSetDict):
        # 模型加载属性
        self.start_epoch = model_app_dict['start_epoch']
        self.device = model_app_dict['device']
        self.save_dir = model_app_dict['save_dir']
        self.log_dir = model_app_dict['log_dir']
        self.load_dir = model_app_dict['load_dir']
        self.test_dir = model_app_dict['test_dir']
        self.phase = model_app_dict['phase']
        self.total_epoch = model_app_dict['total_epoch']
        os.makedirs(self.save_dir, exist_ok=True)
        if self.start_epoch == 0 and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)                               # 清空 log_dir 下的文件（如果存在）
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.netG = model_app_dict["netG"].to(self.device)  # ResnetGenerator(input_nc=2,output_nc=1,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6).to(self.device)
        self.netD = model_app_dict["netD"].to(self.device)  # Discriminator(self.device).to(self.device)

        # 优化器属性
        self.optimG = optim.Adam(self.netG.parameters(), lr=model_app_dict['learning_rate'], betas=(0.9, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=model_app_dict['learning_rate'], betas=(0.9, 0.999))
        self.optimGscheduler = lr_scheduler.StepLR(self.optimG, step_size=model_app_dict['Gscheduler_step_size'], gamma=0.1)
        self.optimDscheduler = lr_scheduler.StepLR(self.optimD, step_size=model_app_dict['Dscheduler_step_size'], gamma=0.1)



        # 定义损失函数
        self.lossD = nn.BCEWithLogitsLoss()  # 判别器损失
        self.lossG = nn.MSELoss()  # 生成器的MSE损失
        self.lossGS = nn.CosineSimilarity(dim=1, eps=1e-08)  # 余弦相似度损失
        self.lossMsSSIM = loss.MS_SSIM_L1_LOSS(self.device)  # MS-SSIM + L1损失
        self.lossL1 = nn.L1Loss()  # L1损失
        self.slic_block_num = model_app_dict['slic_block_num']# 超像素分割的数量
        self.n_segments = 100
        self.c = 100


        # 数据集定义
        self.train_batch_size  = model_app_dict['train_batch_size']
        self.val_batch_size = model_app_dict['val_batch_size']
        self.test_batch_size = model_app_dict['test_batch_size']
        self.train_loader = dataSetDict['train_loader']
        self.val_loader = dataSetDict['val_loader']
        self.test_loader = dataSetDict['test_loader']

        # 初始化训练记录
        self.lossD_list = []
        self.lossG_list = []
        self.lossMSE_list = []
        self.lossD_mean = []
        self.lossG_mean = []
        self.lossMSE_mean = []
        self.val_loss_mean = []




    def _get_image_label(self,inputs):
        one_hot_labelsF = ohe_vector_from_labels(torch.tensor([0] * self.train_batch_size).to(self.device), 2)
        one_hot_labelsR = ohe_vector_from_labels(torch.tensor([1] * self.train_batch_size).to(self.device), 2)

        image_one_hot_labelsF = one_hot_labelsF[:, :, None, None]
        image_one_hot_labelsR = one_hot_labelsR[:, :, None, None]

        # 将标签扩展为与图像相同的空间尺寸
        image_one_hot_labelsF = image_one_hot_labelsF.repeat(1, 1, inputs.shape[2], inputs.shape[3])
        image_one_hot_labelsR = image_one_hot_labelsR.repeat(1, 1, inputs.shape[2], inputs.shape[3])

        return image_one_hot_labelsF, image_one_hot_labelsR

    def _train_discriminator(self, inputs, targets,image_one_hot_labelsF,image_one_hot_labelsR):

        self.netD.zero_grad()
        self.optimD.zero_grad()
        [fake, _] = self.netG(inputs)
        fake_image_and_labels = concat_vectors(fake, image_one_hot_labelsF)
        real_image_and_labels = concat_vectors(targets, image_one_hot_labelsR)

        predD_fake = self.netD(fake_image_and_labels.detach())
        predD_real = self.netD(real_image_and_labels)

        lossD_fake = self.lossD(predD_fake, torch.zeros_like(predD_fake))
        lossD_real = self.lossD(predD_real, torch.ones_like(predD_real))

        lossDT = (lossD_fake + lossD_real) / 2

        lossDT.backward(retain_graph=True)
        self.optimD.step()

        self.lossD_list += [lossDT.data.cpu().numpy() / inputs.shape[0]]

    def _train_generator(self, inputs, targets,up_sampled,image_one_hot_labelsR,number):

        self.netG.zero_grad()
        self.optimG.zero_grad()

        [fake, _] = self.netG(inputs)

        fake_image_and_labels = concat_vectors(fake, image_one_hot_labelsR)
        predD_fake = self.netD(fake_image_and_labels)

        if self.phase == 'first':
            lossG = self.lossD(predD_fake, torch.ones_like(predD_fake))
            lossM = self.lossG(fake, targets)
            g_fake = gradient_img(fake, self.device)
            g_fake = torch.nan_to_num(g_fake)  # ,nan=0,posinf=100,neginf=100)
            g_up_sampled = gradient_img(up_sampled, self.device)
            g_up_sampled = torch.nan_to_num(g_up_sampled)  # ,nan=0,posinf=100,neginf=100)
            lossGS = self.lossGS(g_up_sampled.double(), g_fake.double())
            lossGS = self.lossGS(g_up_sampled, g_fake)
            lossGS = 1 - torch.mean(lossGS)
            lossTV = tvloss(fake)

            lossT = 10 * lossG + 1 * lossM + 10 * lossTV + 10 * lossGS  # + lossTV

        else:
            lossG = self.lossD(predD_fake, torch.ones_like(predD_fake))
            # print('lossG: ',lossG)
            lossSSIM = self.lossMsSSIM(fake, targets)
            lossL1 = self.lossG(fake, targets)

            lossE = self.lossL1(power_loss(up_sampled.squeeze(1), self.device),
                                power_loss(fake.squeeze(1), self.device))

            lossTV = tvloss(fake)
            # I donot get the super pixels of fake only the x sparse
            segmentsi = torch.zeros(self.train_batch_size, self.c)
            segmentf = torch.zeros(self.train_batch_size, self.c)
            for i in range(self.train_batch_size):
                inputs = inputs.detach().cpu()
                si = slic(inputs[i, 2, :, :], n_segments=self.n_segments, sigma=5, channel_axis=None)
                for n, j in enumerate(np.unique(si)):
                    x, y = np.where(si == j)
                    xm, ym = np.unravel_index(np.argmax(inputs[i, 2, :, :][x, y], axis=None),
                                              inputs[i, 2, :, :].shape)
                    segmentsi[i][n] = inputs[i, 2, :, :][xm, ym]

                    segmentf[i][n] = fake[i, 0, xm, ym]
            lossC = self.lossL1(segmentsi.to(self.device), segmentf.to(self.device))

            lossT = lossG + 100 * lossL1 + 0.001 * lossTV + 84 * lossSSIM + lossE + lossC  # + lossC #10*lossL1#+  lossE +  lossTV + lossC

        lossT.backward()
        self.optimG.step()
        self.lossG_list += [lossT.data.cpu().numpy() / inputs.shape[0]]

        with torch.no_grad():
            loss = self.lossG(fake, targets)
            self.lossMSE_list += [loss.data.cpu().numpy()]
            if number % 100 == 0:
                print(' number: ', number, ' MSE: ', np.mean(self.lossMSE_list))

    def validate(self):
        with torch.no_grad():
            self.netD.eval()
            self.netG.eval()
            self.val_loss = []
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                [fake, _] = self.netG(inputs)
                v_loss = self.lossG(fake, targets)
                self.val_loss += [v_loss.item()]

            val = np.mean(np.array(self.val_loss))
            return val

    def train(self):

        # 初始化最佳损失和模型索引
        save_interval = 1
        best_loss = 100
        # 如果从中间epoch开始，加载检查点
        if self.start_epoch != 0:
            checkpoint_path = os.path.join(self.load_dir, f"checkpoint_REMGAN_epoch_{self.start_epoch}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.netG.load_state_dict(checkpoint['netG_state_dict'])
                self.netD.load_state_dict(checkpoint['netD_state_dict'])
                self.optimG.load_state_dict(checkpoint['optimG_state_dict'])
                self.optimD.load_state_dict(checkpoint['optimD_state_dict'])
                self.optimGscheduler.load_state_dict(checkpoint['schedulerG_state_dict'])
                self.optimDscheduler.load_state_dict(checkpoint['schedulerD_state_dict'])
                best_loss = checkpoint.get('best_loss', best_loss)
                print(f"Loaded REMGAN checkpoint from epoch {self.start_epoch}")
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")


        # Dense training
        for epoch in range(self.start_epoch, self.total_epoch):
            train_progress = tqdm(
                self.train_loader,
                desc=f'Epoch {epoch + 1}/{self.total_epoch}',
                leave=True,
                dynamic_ncols=True
            )

            print(f'Epoch {epoch}: Generator LR = {self.optimG.param_groups[0]["lr"]}, 'f'Discriminator LR = {self.optimD.param_groups[0]["lr"]}')
            # 清空每轮的损失列表
            self.lossD_list = []
            self.lossG_list = []
            self.lossMSE_list = []

            for number, (inputs, targets) in enumerate(train_progress):


                inputs, targets = inputs.to(self.device), targets.to(self.device)
                up_sampled = inputs[:, -3, :, :].unsqueeze(1)


                self.netG.train()
                self.netD.train()

                image_one_hot_labelsF, image_one_hot_labelsR = self._get_image_label(inputs)

                # Train the Discriminator
                self._train_discriminator(inputs, targets,image_one_hot_labelsF,image_one_hot_labelsR)
                # Train Generator
                self._train_generator(inputs, targets,up_sampled,image_one_hot_labelsR,number)

            # 计算 epoch 平均损失
            avg_lossD = np.mean(self.lossD_list)
            avg_lossG = np.mean(self.lossG_list)
            avg_lossMSE = np.mean(self.lossMSE_list)
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Discriminator', avg_lossD, epoch)
            self.writer.add_scalar('Loss/Generator', avg_lossG, epoch)
            self.writer.add_scalar('Loss/MSE', avg_lossMSE, epoch)

            print(f"Epoch {epoch} - Mean D loss: {avg_lossD:.6f}, Mean G loss: {avg_lossG:.6f}, Mean MSE: {avg_lossMSE:.6f}")
            self.lossD_mean.append(avg_lossD)
            self.lossG_mean.append(avg_lossG)
            self.lossMSE_mean.append(avg_lossMSE)

            # 需要补一个调度器调节
            self.optimGscheduler.step()
            self.optimDscheduler.step()

            # 验证循环
            val_loss = self.validate()
            self.val_loss_mean.append(val_loss)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"Saving best model with validation MSE: {val_loss:.6f}")
                # 同时保存检查点
                checkpoint = {
                    'epoch': epoch,
                    'netG_state_dict': self.netG.state_dict(),
                    'netD_state_dict': self.netD.state_dict(),
                    'optimG_state_dict': self.optimG.state_dict(),
                    'optimD_state_dict': self.optimD.state_dict(),
                    'schedulerG_state_dict': self.optimGscheduler.state_dict(),
                    'schedulerD_state_dict': self.optimDscheduler.state_dict(),
                    'best_loss': best_loss
                }
                torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_REMGAN_best.pth"))

            # 定期保存检查点
            if epoch % save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'netG_state_dict': self.netG.state_dict(),
                    'netD_state_dict': self.netD.state_dict(),
                    'optimG_state_dict': self.optimG.state_dict(),
                    'optimD_state_dict': self.optimD.state_dict(),
                    'schedulerG_state_dict': self.optimGscheduler.state_dict(),
                    'schedulerD_state_dict': self.optimDscheduler.state_dict(),
                    'best_loss': best_loss
                }
                torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_REMGAN_epoch_{epoch}.pth"))
                print(f"Saved checkpoint at epoch {epoch}")
    def test(self):
        """Test method for REMGAN that loads the best weights and performs testing"""
        print("Predicting on test set with best weights")

        # 加载最佳检查点
        best_checkpoint_path = os.path.join(self.load_dir, "checkpoint_REMGAN_best.pth")
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
            self.netG.load_state_dict(checkpoint['netG_state_dict'])
            self.netD.load_state_dict(checkpoint['netD_state_dict'])
            print(
                f"Loaded best model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['best_loss']:.6f}")
        else:
            print("Warning: Best checkpoint not found. Using current model weights.")

        # 设置模型为评估模式
        self.netG.eval()
        self.netD.eval()

        # 初始化指标
        total_samples = 0
        total_mse = 0.0
        total_energy = 0.0
        total_ssim = 0.0
        total_psnr = 0.0

        # 创建保存目录
        test_dir = os.path.join(self.test_dir, "test_results")
        os.makedirs(test_dir, exist_ok=True)
        # 用于绘制曲线的列表
        batch_nmse_losses = []
        batch_ssim_losses = []
        batch_psnr_losses = []
        batch_indices = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.test_loader, desc="Testing", ncols=100, leave=False)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs[:, :3, :, :]
                # 生成器前向传播
                fake, _ = self.netG(inputs)

                batch_size = inputs.size(0)
                total_samples += batch_size

                # 计算MSE
                mse_batch = self.lossG(fake, targets)
                total_mse += mse_batch.item() * batch_size

                # 计算能量（目标的L2范数平方）
                energy_batch = torch.sum(targets ** 2) / batch_size
                total_energy += energy_batch.item() * batch_size

                # 计算NMSE
                if energy_batch.item() == 0:
                    nmse_loss_value = 0.0 if mse_batch.item() == 0 else float('inf')
                else:
                    nmse_loss_value = mse_batch.item() / energy_batch.item()

                # 计算SSIM
                ssim_batch = ssim(fake, targets)
                total_ssim += ssim_batch.item() * batch_size

                # 计算PSNR
                psnr_batch = psnr(fake, targets)
                total_psnr += psnr_batch.item() * batch_size

                # 记录批次指标
                batch_nmse_losses.append(nmse_loss_value)
                batch_ssim_losses.append(ssim_batch.item())
                batch_psnr_losses.append(psnr_batch.item())
                batch_indices.append(batch_idx)

                # 保存一些示例图像
                if batch_idx < 5:  # 只保存前5个批次的图像
                    self._save_comparison_images(fake, targets, batch_idx, test_dir)

                # 记录到TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Test/NMSE_batch', nmse_loss_value, batch_idx)
                    self.writer.add_scalar('Test/SSIM_batch', ssim_batch.item(), batch_idx)
                    self.writer.add_scalar('Test/PSNR_batch', psnr_batch.item(), batch_idx)

        # 计算平均指标
        avg_mse = total_mse / total_samples if total_samples > 0 else 0
        avg_rmse = math.sqrt(avg_mse)

        if total_energy == 0:
            avg_nmse = 0.0 if total_mse == 0 else float('inf')
        else:
            avg_nmse = total_mse / total_energy

        avg_ssim = total_ssim / total_samples if total_samples > 0 else 0
        avg_psnr = total_psnr / total_samples if total_samples > 0 else 0

        # 记录平均指标到TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Test/NMSE', avg_nmse, 0)
            self.writer.add_scalar('Test/RMSE', avg_rmse, 0)
            self.writer.add_scalar('Test/SSIM', avg_ssim, 0)
            self.writer.add_scalar('Test/PSNR', avg_psnr, 0)

        # 绘制指标曲线
        self._plot_test_metrics(batch_indices, batch_nmse_losses, batch_ssim_losses, batch_psnr_losses, test_dir)

        # 打印结果
        print(f"Test Results:")
        print(f"NMSE: {avg_nmse:.6f}")
        print(f"RMSE: {avg_rmse:.6f}")
        print(f"SSIM: {avg_ssim:.6f}")
        print(f"PSNR: {avg_psnr:.6f}")

        # 保存结果到文件
        with open(os.path.join(test_dir, "test_results.txt"), "w") as f:
            f.write(f"NMSE: {avg_nmse:.6f}\n")
            f.write(f"RMSE: {avg_rmse:.6f}\n")
            f.write(f"SSIM: {avg_ssim:.6f}\n")
            f.write(f"PSNR: {avg_psnr:.6f}\n")

        return avg_nmse, avg_rmse, avg_ssim, avg_psnr

    def _save_comparison_images(self, fake, targets, batch_idx, save_dir):
        """保存比较图像"""
        # 选择批次中的第一个样本
        fake_img = fake[0].cpu().squeeze()
        target_img = targets[0].cpu().squeeze()

        # 创建比较图像
        comparison = torch.stack([target_img, fake_img], dim=0)

        # 保存图像
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(target_img, cmap='viridis')
        plt.title('Target')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(fake_img, cmap='viridis')
        plt.title('Generated')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_batch_{batch_idx}.png"))
        plt.close()

        # 保存到TensorBoard
        if self.writer is not None:
            grid = torchvision.utils.make_grid(comparison.unsqueeze(1), nrow=2, normalize=True)
            self.writer.add_image(f'Test/Comparison_batch_{batch_idx}', grid, 0)

    def _plot_test_metrics(self, batch_indices, nmse_losses, ssim_losses, psnr_losses, save_dir):
        """绘制测试指标曲线"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(batch_indices, nmse_losses, 'b-o')
        plt.title('NMSE per Batch')
        plt.xlabel('Batch Index')
        plt.ylabel('NMSE')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(batch_indices, ssim_losses, 'r-o')
        plt.title('SSIM per Batch')
        plt.xlabel('Batch Index')
        plt.ylabel('SSIM')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(batch_indices, psnr_losses, 'g-o')
        plt.title('PSNR per Batch')
        plt.xlabel('Batch Index')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "test_metrics.png"))
        plt.close()

        # 保存到TensorBoard
        if self.writer is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].plot(batch_indices, nmse_losses, 'b-o')
            axes[0].set_title('NMSE per Batch')
            axes[0].set_xlabel('Batch Index')
            axes[0].set_ylabel('NMSE')
            axes[0].grid(True)

            axes[1].plot(batch_indices, ssim_losses, 'r-o')
            axes[1].set_title('SSIM per Batch')
            axes[1].set_xlabel('Batch Index')
            axes[1].set_ylabel('SSIM')
            axes[1].grid(True)

            axes[2].plot(batch_indices, psnr_losses, 'g-o')
            axes[2].set_title('PSNR per Batch')
            axes[2].set_xlabel('Batch Index')
            axes[2].set_ylabel('PSNR (dB)')
            axes[2].grid(True)

            plt.tight_layout()
            self.writer.add_figure('Test/Metrics', fig, 0)
            plt.close()


