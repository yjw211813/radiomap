import torch
from model.rem_gan.loss import MS_SSIM_L1_LOSS

import torch.nn as nn

from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import glob
import shutil
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import torchvision
import math
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F




class REM_GAN_app():
    def __init__(self, model_app_dict,dataSetDict):

        # 模型加载属性
        self.start_epoch = model_app_dict['start_epoch']
        self.device = model_app_dict['device']
        self.save_dir = model_app_dict['save_dir']
        self.log_dir = model_app_dict['log_dir']
        self.load_dir = model_app_dict['load_dir']

        os.makedirs(self.save_dir, exist_ok=True)
        if self.start_epoch == 0 and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)                               # 清空 log_dir 下的文件（如果存在）
        os.makedirs(self.log_dir, exist_ok=True)


        # 模型定义
        self.n_segments = 100    # 超像素分割的数量
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
        self.lossMsSSIM = MS_SSIM_L1_LOSS()  # MS-SSIM + L1损失
        self.lossL1 = nn.L1Loss()  # L1损失

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



       #  self.log_dir = log_dir
       #
       #  self.step_size = step_size
       #  self.model_save_dir = model_save_dir
       #
       #  self.device = device
       #  self.num_loss_samples = num_loss_samples

       #  self.writer = SummaryWriter(log_dir=self.log_dir)

    def ohe_vector_from_labels(self,labels, n_classes):
        return F.one_hot(labels, num_classes=n_classes)

    def concat_vectors(self,x, y):
        combined = torch.cat((x.float(), y.float()), 1)
        return combined


    def get_GAN_confition(self,H,W):
        # 生成 batch_sz 长度的 独热码
        one_hot_labelsF = self.ohe_vector_from_labels(torch.tensor([0] * self.train_batch_size).to(self.device), 2)
        one_hot_labelsR = self.ohe_vector_from_labels(torch.tensor([1] * self.train_batch_size).to(self.device), 2)

        image_one_hot_labelsF = one_hot_labelsF[:, :, None, None]
        image_one_hot_labelsR = one_hot_labelsR[:, :, None, None]

        self.netD.zero_grad()
        self.optimD.zero_grad()

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

    def evaluate(self,model,val_loader,epoch,WNetPhase = "firstU" , targetType = "dense"):
        print("evaluating on val set")

    def train(self,model,train_loader,val_loader,total_epoch,WNetPhase = "firstU" , targetType = "dense"):

        for epoch in range(self.start_epoch, total_epoch):
            train_progress = tqdm(
                train_loader,
                desc=f'Epoch {epoch + 1}/{total_epoch}',
                leave=True,
                dynamic_ncols=True
            )

            # 训练阶段
            model.train()
            # 两种训练模式:
            for inputs, targets in train_progress:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                up_sampled = inputs[:, 3, :, :].unsqueeze(1)

                self.netG.train()
                self.netD.train()
                ############################
                # Train the Discriminator  #
                ############################
                lossDT, fake = self._train_discriminator(inps, gts)

                self.lossD_list.append(lossDT.data.cpu().numpy() / inps.shape[0])
















        # 需要补一个调度器调节
        self.optimGscheduler.step()
        self.optimDscheduler.step()


            ############################
            # Train the Discriminator  #
            ############################


        print("training on train set")

    def predict(self, model, load_epoch, test_loader, val_dir,WNetPhase="secondU", targetType = "dense"):

        print("predicting on test set")

    def create_horizontal_comparison(self, outputs, targets, batch_idx, val_dir):
        # 确保输入是numpy数组且形状正确
        if torch.is_tensor(outputs):
            outputs = outputs.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()

        # 移除通道维度 (batch_size, 1, H, W) -> (batch_size, H, W)
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)

        num_batches = outputs.shape[0]

        # 创建一个大图像，包含所有批次的对比
        fig, axes = plt.subplots(2, num_batches, figsize=(5 * num_batches, 10))

        # 处理只有1个批次的情况
        if num_batches == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_batches):
            # 获取当前批次的target和output
            target_img = targets[i]
            output_img = outputs[i]

            # 显示target
            ax = axes[0, i]
            im = ax.imshow(target_img, cmap='jet')
            ax.set_title(f"Target (Sample {i})")
            ax.axis('off')

            # 显示output
            ax = axes[1, i]
            im = ax.imshow(output_img, cmap='jet')
            ax.set_title(f"Output (Sample {i})")
            ax.axis('off')

        # 添加一个共享的颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(val_dir, f"batch_{batch_idx}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def save_best_checkpoint(self,model, optimizer, scheduler, epoch, current_loss, best_loss,
                             save_dir, phase_name, delete_previous=True):
        """
        保存最优权重检查点，并可选择删除之前的最优权重

        Args:
            model: 模型实例
            optimizer: 优化器实例
            scheduler: 学习率调度器实例
            epoch: 当前epoch
            current_loss: 当前验证损失
            best_loss: 历史最佳损失
            save_dir: 保存目录
            phase_name: 阶段名称
            delete_previous: 是否删除之前的最优权重文件

        Returns:
            new_best_loss: 更新后的最佳损失值
            is_best: 是否是最佳模型
        """
        is_best = current_loss < best_loss
        new_best_loss = min(current_loss, best_loss)

        if is_best:
            # 如果设置为删除之前的最优权重，先查找并删除
            if delete_previous:
                pattern = os.path.join(save_dir, f"best_checkpoint_{phase_name}_epoch_*.pth")
                previous_best_files = glob.glob(pattern)
                for file_path in previous_best_files:
                    try:
                        os.remove(file_path)
                        print(f"Deleted previous best checkpoint: {os.path.basename(file_path)}")
                    except OSError as e:
                        print(f"Error deleting file {file_path}: {e}")

            # 创建检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': new_best_loss,
                'val_loss': current_loss
            }

            # 保存新的最优权重
            checkpoint_path = os.path.join(save_dir, f"best_checkpoint_{phase_name}_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint at epoch {epoch + 1} with loss: {current_loss:.6f}")

        return new_best_loss, is_best

    def load_best_checkpoint(self, model, WNetPhase):
        """
        加载指定阶段的最优权重检查点到模型中

        Args:
            model: 要加载权重的模型实例
            phase_name: 阶段名称（如"firstU"、"secondU"）

        Returns:
            model: 加载了最优权重的模型
            best_loss: 最佳损失值
            epoch: 最佳权重对应的epoch
        """
        # 构建最优权重文件的搜索模式
        pattern = os.path.join(self.model_save_dir, f"best_checkpoint_{WNetPhase}_epoch_*.pth")
        best_checkpoint_files = glob.glob(pattern)

        if not best_checkpoint_files:
            print(f"警告: 未找到{WNetPhase}阶段的最优权重文件")
            return model, float('inf'), 0

        # 按epoch排序，选择最新的最优权重文件
        # 从文件名中提取epoch号并排序
        def extract_epoch(filename):
            match = re.search(r'epoch_(\d+)\.pth$', filename)
            return int(match.group(1)) if match else 0

        best_checkpoint_files.sort(key=extract_epoch, reverse=True)
        latest_best_checkpoint = best_checkpoint_files[0]

        # 加载检查点
        try:
            checkpoint = torch.load(latest_best_checkpoint, weights_only=True, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

            best_loss = checkpoint.get('best_loss', float('inf'))
            epoch = checkpoint.get('epoch', 0)

            print(f"成功加载{WNetPhase}阶段的最优权重 (epoch {epoch}, loss: {best_loss:.6f})")
            return model, best_loss, epoch

        except Exception as e:
            print(f"加载最优权重时出错: {e}")
            return model, float('inf'), 0




if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    model = RadioWNet(phase="firstU")
    model.to(device)