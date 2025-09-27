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

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torch.optim import lr_scheduler
import torchvision
import shutil
from model.rem_gan import modules, loss
import torch.nn.functional as F
import torch.nn as nn
from model.rem_gan.EncoderModels import ResnetGenerator, Discriminator
from data.lib.seer_loader import RadioMapSeerLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.segmentation import slic
import scipy.stats as stats

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
    def __init__(self, netD, netG, model_app_dict, dataSetDict):
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

        self.netG = netG.to(self.device)  # ResnetGenerator(input_nc=2,output_nc=1,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6).to(self.device)
        self.netD = netD.to(self.device)  # Discriminator(self.device).to(self.device)

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
                up_sampled = inputs[:, 3, :, :]
                inputs = inputs[:, :3, :, :]
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
                up_sampled = inputs[:, 3, :, :].unsqueeze(1)
                inputs = inputs[:, :3, :, :]

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


if __name__ == '__main__':
    ########################
    # Load dataset         #
    ########################
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    setup = 1  # REVISE index of setup
    setups = ['uniform', 'twoside', 'nonuniform']
    setup_name = setups[setup - 1]

    simuSetDict = {
        "ind1": 0,  # 起始索引
        "ind2": 0,  # 末尾索引
        "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",  # 数据集文件夹
        "numTx": 80,  # 信源数量设定
        "thresh": 0.2,  # 环境噪声
        "simulation": "rand",  # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
        "carsSimul": "no",  # 是否开启小车作为仿真
        "carsInput": "no",  # 是否将小车图作为模型输入
        "IRT2maxW": 0.3,  # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
        "cityMap": "complete",  # 是否输入完全的城市地图
        "missing": 1,  # 地图缺失号码
        "fix_samples": 0,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
        "num_samples_low": 655,  # 最低采样数
        "num_samples_high": 655 * 10,  # 最高采样数
        "inter_flag": False,  # 看是否需要插值图像
        "scale256_flag": True,  # 取值范围是否为0 - 255
        "sample_flag": True,  # 是否有采样输入
        "loss_samples_flag": False,  # 是否定义loss为稀疏采样loss
        "formula_flag": True
    }

    if setup == 1:
        simuSetDict["fix_samples"] = 655
    elif setup == 2:
        simuSetDict["fix_samples"] = 1
    else:
        simuSetDict["fix_samples"] = 0

    train_batch_size = 30  # 批次大小
    val_batch_size = 30
    test_batch_size = 30  # 批次大小1
    # 加载数据集
    Radio_train = RadioMapSeerLoader(simuSetDict, phase="train")
    Radio_val = RadioMapSeerLoader(simuSetDict, phase="val")
    Radio_test = RadioMapSeerLoader(simuSetDict, phase="test")

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=train_batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(Radio_val, batch_size=val_batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(Radio_test, batch_size=test_batch_size, shuffle=True, num_workers=4)
    }
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # 设置保存目录
    log_dir = r'/home/code/radio_map_construction/runs/model_log/REM_GAN/'
    model_load_dir = r"/home/code/radio_map_construction/runs/model_pth/REM_GAN/"
    model_save_dir = r"/home/code/radio_map_construction/runs/model_pth/REM_GAN/"
    test_dir = r"/home/code/radio_map_construction/runs/test_results/REM_GAN/"

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print("REM_GAN")
    netG = modules.RadioWNet(phase="firstU")
    netD = Discriminator()
    # 配置模型应用字典
    model_app_dict = {
        'start_epoch': 0,
        'device': device,
        'save_dir': model_save_dir,
        'log_dir': log_dir,
        'load_dir': model_load_dir,
        'test_dir': test_dir,
        'phase': 'first',
        'total_epoch': 400,
        'netG': netG,
        'netD': netD,
        'learning_rate': 0.001,
        'Gscheduler_step_size': 25,
        'Dscheduler_step_size': 25,
        'slic_block_num': 100,
        'train_batch_size': train_batch_size,
        'val_batch_size': val_batch_size,
        'test_batch_size': test_batch_size
    }

    # 配置数据集字典
    dataSetDict = {
        'trainset': Radio_train,
        'valset': Radio_val,
        'testset': Radio_test,
        'train_batch_size': train_batch_size,
        'val_batch_size': val_batch_size,
        'test_batch_size': test_batch_size
    }
    # 初始化REM_GAN应用
    rem_gan_app = REM_GAN_app(netD, netG,model_app_dict, dataSetDict)

    # 训练模型
    print("开始训练REM-GAN模型...")
    rem_gan_app.train()
    # 测试模型
    print("开始测试REM-GAN模型...")
    # rem_gan_app.test()
    # print("REM-GAN 训练和测试完成")

