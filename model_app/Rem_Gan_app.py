import torch
from model.rem_gan.loss import MS_SSIM_L1_LOSS
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import torchvision
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
from skimage.segmentation import slic


# 超参数配置
TV_WEIGHT = 1e-7# 总变差损失的权重

class REM_GAN_app():
    def __init__(self, model_app_dict,dataSetDict):

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

        # 模型定义

        self.netG = model_app_dict['netG'].to(self.device)
        self.netD = model_app_dict['netD'].to(self.device)


        # 优化器属性
        self.optimG = optim.Adam(self.netG.parameters(), lr=model_app_dict['learning_rate'], betas=(0.9, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=model_app_dict['learning_rate'], betas=(0.9, 0.999))
        self.optimGscheduler = lr_scheduler.StepLR(self.optimG, step_size=model_app_dict['Gscheduler_step_size'], gamma=0.1)
        self.optimDscheduler = lr_scheduler.StepLR(self.optimD, step_size=model_app_dict['Dscheduler_step_size'], gamma=0.1)



        # 定义损失函数
        self.lossD = nn.BCEWithLogitsLoss()  # 判别器损失
        self.lossG = nn.MSELoss()  # 生成器的MSE损失
        self.lossGS = nn.CosineSimilarity(dim=1, eps=1e-08)  # 余弦相似度损失
        self.lossMsSSIM = MS_SSIM_L1_LOSS(self.device)  # MS-SSIM + L1损失
        self.lossL1 = nn.L1Loss()  # L1损失
        self.slic_block_num = model_app_dict['slic_block_num']# 超像素分割的数量



        # 数据集定义
        self.train_batch_size  = model_app_dict['train_batch_size']
        self.val_batch_size = model_app_dict['val_batch_size']
        self.test_batch_size = model_app_dict['test_batch_size']
        self.train_loader = DataLoader(dataSetDict['trainset'], batch_size=dataSetDict['train_batch_size'], shuffle=False, num_workers=2)
        self.val_loader = DataLoader(dataSetDict['valset'], batch_size=dataSetDict['val_batch_size'], shuffle=False, num_workers=2)
        self.test_loader = DataLoader(dataSetDict['testset'], batch_size=dataSetDict['test_batch_size'], shuffle=False, num_workers=2)

        # 初始化训练记录
        self.lossD_list = []
        self.lossG_list = []
        self.lossMSE_list = []
        self.lossD_mean = []
        self.lossG_mean = []
        self.lossMSE_mean = []
        self.val_loss_mean = []

    def ohe_vector_from_labels(self,labels, n_classes):
        return F.one_hot(labels, num_classes=n_classes)

    def concat_vectors(self,x, y):
        combined = torch.cat((x.float(), y.float()), 1)
        return combined
    def tv_loss(self, y_hat):
        """计算总变差损失，用于平滑生成的图像"""
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        return TV_WEIGHT * (diff_i + diff_j)

    def power_spectrum_loss(self, images, E=100):
        """计算功率谱损失，用于保持频谱特性"""
        # 将图像从GPU移动到CPU并转换为numpy数组
        image = images.detach().cpu().numpy().astype(int)
        npix = images.shape[1]# 获取图像尺寸

        # 计算傅里叶变换和功率谱
        fourier_image = np.fft.fftn(image) # 对图像进行N维傅里叶变换
        fourier_amplitudes = np.abs(fourier_image) ** 2# 计算功率谱（振幅的平方）

        # 计算频率
        kfreq = np.fft.fftfreq(npix) * npix # 获取频率值并缩放到像素单位
        kfreq2D = np.meshgrid(kfreq, kfreq)# 创建二维频率网格
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)# 计算每个频率点的模

        # 扁平化处理以便进行统计分析
        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.reshape(images.shape[0], -1)

        # 计算分箱统计 - 将频率按半径分箱并计算每个箱内的平均功率
        kbins = np.arange(0.5, npix // 2 + 1, 1.)# 创建分箱边界
        kvals = 0.5 * (kbins[1:] + kbins[:-1]) # 计算每个箱的中心值

        # 对功率谱按频率模数进行分箱统计
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                             statistic="mean",
                                             bins=kbins)
        # 考虑二维空间的面积元素（环形区域面积）
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

        # 获取前E个最重要的频率分量（功率最大的频率区间）
        ind = np.argpartition(Abins, E)
        return torch.FloatTensor(ind).to(self.device)

    def compute_gradients(self, img):
        """计算批次图像的梯度，用于梯度相似度损失"""

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

    def get_GAN_confition(self,H,W):
        # 生成 batch_sz 长度的 独热码
        one_hot_labelsF = self.ohe_vector_from_labels(torch.tensor([0] * self.train_batch_size).to(self.device), 2)
        one_hot_labelsR = self.ohe_vector_from_labels(torch.tensor([1] * self.train_batch_size).to(self.device), 2)

        image_one_hot_labelsF = one_hot_labelsF[:, :, None, None]
        image_one_hot_labelsR = one_hot_labelsR[:, :, None, None]

        # 将标签扩展为与图像相同的空间尺寸
        image_one_hot_labelsF = image_one_hot_labelsF.repeat(1, 1, H, W)
        image_one_hot_labelsR = image_one_hot_labelsR.repeat(1, 1, H, W)

        return image_one_hot_labelsF, image_one_hot_labelsR

    def _train_discriminator(self, inputs, targets):
        """训练判别器"""
        self.netD.zero_grad()
        self.optimD.zero_grad()

        image_one_hot_labelsF, image_one_hot_labelsR = self.get_GAN_confition(inputs.shape[2], inputs.shape[3])

        # 生成假图像
        [fake, _] = self.netG(inputs)
        fake_image_and_labels = self.concat_vectors(fake, image_one_hot_labelsF)
        real_image_and_labels = self.concat_vectors(targets, image_one_hot_labelsR)

        # 判别器预测
        predD_fake = self.netD(fake_image_and_labels.detach())
        predD_real = self.netD(real_image_and_labels)

        # 计算损失
        lossD_fake = self.lossD(predD_fake, torch.zeros_like(predD_fake))
        lossD_real = self.lossD(predD_real, torch.ones_like(predD_real))
        lossDT = (lossD_fake + lossD_real) / 2

        # 反向传播
        lossDT.backward(retain_graph=True)
        self.optimD.step()

        return lossDT, fake

    def _train_generator(self, inputs, targets,up_sampled):

        self.netG.zero_grad()
        self.optimG.zero_grad()

        [fake, _] = self.netG(inputs)
        image_one_hot_labelsF, image_one_hot_labelsR = self.get_GAN_confition(inputs.shape[2], inputs.shape[3])
        fake_image_and_labels = self.concat_vectors(fake, image_one_hot_labelsR)
        predD_fake = self.netD(fake_image_and_labels)

        if self.phase == 'first':
            lossT = self._get_generator_loss_first_phase(predD_fake, fake, targets, up_sampled)

        else:
            lossT = self._get_generator_loss_second_phase(predD_fake, fake, inputs, targets, up_sampled)

        lossT.backward()
        self.optimG.step()

        return lossT

    def _get_generator_loss_first_phase(self,predD_fake, fake, targets,up_sampled):
        lossG = self.lossD(predD_fake, torch.ones_like(predD_fake))
        lossM = self.lossG(fake, targets)
        g_fake = self.compute_gradients(fake)
        g_fake = torch.nan_to_num(g_fake)  # ,nan=0,posinf=100,neginf=100)
        g_up_sampled = self.compute_gradients(up_sampled)
        g_up_sampled = torch.nan_to_num(g_up_sampled)  # ,nan=0,posinf=100,neginf=100)
        lossGS = self.lossGS(g_up_sampled.double(), g_fake.double())

        lossGS = 1 - torch.mean(lossGS)
        lossTV = self.tv_loss(fake)

        lossT = 10 * lossG + 1 * lossM + 10 * lossTV + 10 * lossGS  # + lossTV
        return lossT

    def _get_generator_loss_second_phase(self,predD_fake, fake,inputs, targets,up_sampled):
        lossG = self.lossD(predD_fake, torch.ones_like(predD_fake))
        lossSSIM = self.lossMsSSIM(fake, targets)
        lossL1 = self.lossG(fake, targets)
        lossE = self.lossL1(self.power_spectrum_loss(up_sampled.squeeze(1), self.device),
                            self.power_spectrum_loss(fake.squeeze(1), self.device))
        lossTV = self.tv_loss(fake)

        segmentsi = torch.zeros(self.train_batch_size, self.slic_block_num)
        segmentf = torch.zeros(self.train_batch_size, self.slic_block_num)
        for i in range(self.train_batch_size):
            inputs = inputs.detach().cpu()
            si = slic(inputs[i, 2, :, :], n_segments=self.slic_block_num, sigma=5, channel_axis=None)
            for n, j in enumerate(np.unique(si)):
                x, y = np.where(si == j)
                xm, ym = np.unravel_index(np.argmax(inputs[i, 2, :, :][x, y], axis=None),
                                          inputs[i, 2, :, :].shape)
                segmentsi[i][n] = inputs[i, 2, :, :][xm, ym]
                segmentf[i][n] = fake[i, 0, xm, ym]
        lossC = self.lossL1(segmentsi.to(self.device), segmentf.to(self.device))

        lossT = lossG + 100 * lossL1 + 0.001 * lossTV + 84 * lossSSIM + lossE + lossC  # + lossC #10*lossL1#+  lossE +  lossTV + lossC

        return lossT

    def validate(self):
        """验证模型"""
        self.netD.eval()
        self.netG.eval()
        val_losses = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                up_sampled = inputs[:, 3, :, :]
                inputs = inputs[:, :3, :, :]

                fake, _ = self.netG(inputs)
                v_loss = self.lossG(fake, targets)
                val_losses.append(v_loss.item())

        return np.mean(val_losses)


    def train(self):
        # 初始化最佳损失和模型索引
        best_loss = float('inf')
        save_interval = 1

        # 如果从中间epoch开始，加载检查点
        if self.start_epoch != 0:
            checkpoint_path = os.path.join(self.load_dir, f"checkpoint_REMGAN_epoch_{self.start_epoch}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
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

            # 两种训练模式:
            for number,(inputs, targets) in enumerate(train_progress):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                up_sampled = inputs[:, 3, :, :].unsqueeze(1) # 得到公式相关图像 log对数拟合的图像
                inputs = inputs[:, :3, :, :]

                self.netG.train()
                self.netD.train()

                lossDT, fake = self._train_discriminator(inputs, targets) # Train the Discriminator  #
                self.lossD_list.append(lossDT.data.cpu().numpy() / inputs.shape[0])

                lossT = self._train_generator(inputs, targets,up_sampled)# Train Generator   #
                self.lossG_list.append(lossT.data.cpu().numpy() / inputs.shape[0])

                with torch.no_grad():
                    loss = self.lossG(fake,targets)
                    self.lossMSE_list.append(loss.data.cpu().numpy())
                    if number%100 == 0:
                        print("epoch: ",epoch,' number: ',number,' MSE: ',np.mean(self.lossMSE_list))

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
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
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




if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
