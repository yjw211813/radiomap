from data.lib.seer_loader import RadioMapSeerLoader
from torch.utils.data import Dataset, DataLoader
import torch
from model.radioUnet.RadioUnetModel import RadioWNet
from torchsummary import summary
import torch.nn as nn
import copy
import time
from collections import defaultdict
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

# def get_dataset():
#     simuSetDict = {
#         "ind1": 0,  # 起始索引
#         "ind2": 0,  # 末尾索引
#         "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",  # 数据集文件夹
#         "numTx": 80,  # 信源数量设定
#         "thresh": 0.05,  # 环境噪声
#         "simulation": "DPM",  # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
#         "carsSimul": "no",  # 是否开启小车作为仿真
#         "carsInput": "no",  # 是否将小车图作为模型输入
#         "IRT2maxW": 1,  # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
#         "cityMap": "complete",  # 是否输入完全的城市地图
#         "missing": 1,  # 地图缺失号码
#         "fix_samples": 300,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
#         "num_samples_low": 10,  # 最低采样数
#         "num_samples_high": 300,  # 最高采样数
#         "inter_flag": False,  # 看是否需要插值图像
#         "sample_flag": False,
#         "scale256_flag": False,  # 取值范围是否为0 - 255
#         "loss_samples_flag":False,# 是否定义loss为稀疏采样loss
#     }
#     # 加载数据集
#     Radio_train = RadioMapSeerLoader(simuSetDict, phase="train")
#
#
#     Radio_val = RadioMapSeerLoader(phase="val")
#     Radio_test = RadioMapSeerLoader(phase="test")
#
#     image_datasets = {
#         'train': Radio_train, 'val': Radio_val,'test': Radio_test
#     }
#
#     batch_size = 15
#
#     dataloaders = {
#         'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1),
#         'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1)
#     }

class RadioWNet_app():
    def __init__(self, start_epoch, log_dir, step_size, model_save_dir,device,num_loss_samples = 300):
        self.start_epoch = start_epoch
        self.log_dir = log_dir
        self.step_size = step_size
        self.model_save_dir = model_save_dir
        self.device = device
        self.num_loss_samples = num_loss_samples
       # 清空 log_dir 下的文件（如果存在）
        if self.start_epoch == 0 and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def evaluate(self,model,val_loader,epoch,WNetPhase = "firstU" , targetType = "dense"):
        model.to(self.device)
        model.eval()
        criterion = nn.MSELoss()
        loss_temp = 0
        num_samples = 0

        val_batch = 20
        eval_progress = tqdm(
            val_loader,
            desc=f'evaluate {epoch}',
            leave=True,
            dynamic_ncols=True
        )

        with torch.no_grad():
            if targetType == "dense":
                for batch_idx, (inputs, targets) in enumerate(eval_progress):
                    if batch_idx >= val_batch:  # 达到指定batch数后跳出循环
                        break

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    [outputs1, outputs2] = model(inputs)
                    if WNetPhase == "firstU":
                        loss = criterion(outputs1, targets)
                    else:
                        loss = criterion(outputs2, targets)

                    loss_temp += loss.item() * targets.size(0)
                    num_samples += targets.size(0)
                    eval_progress.set_postfix(loss=f'{loss.item():.4f}')

            elif targetType == "sparse":
                for batch_idx, (inputs, targets, loss_samples) in enumerate(eval_progress):
                    if batch_idx >= val_batch:  # 达到指定batch数后跳出循环
                        break

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    loss_samples = loss_samples.to(self.device)

                    [outputs1, outputs2] = model(inputs)

                    if WNetPhase == "firstU":
                        # Only calculate loss on valid samples
                        loss = criterion(loss_samples * outputs1, loss_samples * targets)
                    else:
                        loss = criterion(loss_samples * outputs2, loss_samples * targets)
                    # Normalize by the number of valid samples, not batch size
                    loss = loss * (256 ** 2) / self.num_loss_samples

                    loss_temp += loss.item() * targets.size(0)
                    num_samples += targets.size(0)
                    eval_progress.set_postfix(loss=f'{loss.item():.4f}')

        avg_train_loss = loss_temp / num_samples
        self.writer.add_scalar('Loss/train', avg_train_loss, epoch)

        return avg_train_loss

    def train(self,model,train_loader,val_loader,total_epoch,WNetPhase = "firstU" , targetType = "dense"):
        model.to(self.device)
        eval_interval = 1
        save_interval = 1
        # 中间变量初始化

        best_loss = 1e10

        # 定义优化器

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.1)
        criterion = nn.MSELoss()

        # 从过往权重加载模型
        if self.start_epoch != 0:
            checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_{WNetPhase}_epoch_{self.start_epoch}.pth")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint.get('best_loss', best_loss)
            print(f"Loaded checkpoint from epoch {self.start_epoch}")

        # 使用进度条记录模型
        for epoch in range(self.start_epoch, total_epoch):
            train_progress = tqdm(
                train_loader,
                desc=f'Epoch {epoch + 1}/{total_epoch}',
                leave=True,
                dynamic_ncols=True
            )

            loss_temp = 0
            num_samples = 0
            # 训练阶段
            model.train()

            # 两种训练模式:
            if targetType == "dense":
                for inputs, targets in train_progress:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        [outputs1, outputs2] = model(inputs)
                        if WNetPhase == "firstU":
                            loss = criterion(outputs1, targets)
                        else:
                            loss = criterion(outputs2, targets)

                        loss.backward()
                        optimizer.step()

                        loss_temp += loss.data.cpu().numpy() * targets.size(0)

                    num_samples += inputs.size(0)
                    train_progress.set_postfix(loss=f'{loss.item():.4f}')

            elif targetType == "sparse":
                for inputs, targets,loss_samples in train_progress:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    loss_samples = loss_samples.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        [outputs1, outputs2] = model(inputs)
                        if WNetPhase == "firstU":
                            # Only calculate loss on valid samples
                            loss = criterion(loss_samples * outputs1, loss_samples * targets)
                        else:
                            loss = criterion(loss_samples * outputs2, loss_samples * targets)
                        # Normalize by the number of valid samples, not batch size
                        loss = loss * (256 ** 2) / self.num_loss_samples

                        loss.backward()
                        optimizer.step()

                        loss_temp += loss.data.cpu().numpy() * targets.size(0)

                    num_samples += inputs.size(0)
                    train_progress.set_postfix(loss=f'{loss.item():.4f}')

            # 更新学习率
            scheduler.step()

            avg_train_loss = loss_temp / num_samples
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)


            # 验证阶段
            if (epoch + 1) % eval_interval == 0:
                avg_val_loss = self.evaluate(model, val_loader,epoch, WNetPhase = WNetPhase)
                self.writer.add_scalar('Loss/val', avg_val_loss, epoch)

                # 保存最佳模型
                best_loss, is_best = self.save_best_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    current_loss=avg_val_loss,
                    best_loss=best_loss,
                    save_dir=self.model_save_dir,
                    phase_name=WNetPhase,
                    delete_previous=True  # 设置为True会自动删除之前的最优权重
                )

            # 定期保存检查点
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss
                }
                torch.save(checkpoint,
                           os.path.join(self.model_save_dir, f"checkpoint_{WNetPhase}_epoch_{epoch}.pth"))
                print(f"Saved checkpoint at epoch {epoch}")


    def predict(self, model, load_epoch, test_loader, val_dir,WNetPhase="secondU", targetType = "dense"):

        """Predict method with loss curves as requested"""

        checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_{WNetPhase}_epoch_{load_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        print(f"加载历史数据load_epoch:{load_epoch}成功")
        model.load_state_dict(checkpoint['model_state_dict'])


        model.to(self.device)
        model.eval()
        criterion = nn.MSELoss()
        total_samples = 0
        total_mse = 0.0
        total_energy = 0.0
        total_ssim = 0.0
        total_psnr = 0.0

        os.makedirs(val_dir, exist_ok=True)

        batch_nmse_losses = []
        batch_ssim_losses = []
        batch_psnr_losses = []
        batch_indices = []

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing", ncols=100, leave=False)):

                # 根据targetType解包数据
                if targetType == "dense":
                    inputs, targets = data
                    loss_samples = None
                elif targetType == "sparse":
                    inputs, targets, loss_samples = data
                    loss_samples = loss_samples.to(self.device)
                else:
                    raise ValueError("Invalid targetType. Choose 'dense' or 'sparse'.")

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                [outputs1, outputs2] = model(inputs)
                # 根据WNetPhase选择输出
                if WNetPhase == "firstU":
                    outputs = outputs1
                else:
                    outputs = outputs2

                batch_size = inputs.size(0)
                total_samples += batch_size

                # 计算损失和指标
                if targetType == "dense":
                    mse_batch = criterion(outputs, targets)
                    energy_batch = criterion(targets, torch.zeros_like(targets))
                else:  # sparse
                    # 只计算有效样本的损失
                    mse_batch = criterion(loss_samples * outputs, loss_samples * targets)
                    energy_batch = criterion(loss_samples * targets, torch.zeros_like(targets))
                    # 归一化
                    mse_batch = mse_batch * (256 ** 2) / self.num_loss_samples
                    energy_batch = energy_batch * (256 ** 2) / self.num_loss_samples

                total_mse += mse_batch.item() * batch_size
                total_energy += energy_batch.item() * batch_size

                if energy_batch.item() == 0:
                    nmse_loss_value = 0.0 if mse_batch.item() == 0 else float('inf')
                else:
                    nmse_loss_value = mse_batch.item() / energy_batch.item()


                ssim_batch = ssim(outputs, targets)
                total_ssim += ssim_batch.item() * batch_size

                psnr_batch = psnr(outputs, targets)
                total_psnr += psnr_batch.item() * batch_size

                batch_nmse_losses.append(nmse_loss_value)
                batch_ssim_losses.append(ssim_batch.item())
                batch_psnr_losses.append(psnr_batch.item())
                batch_indices.append(batch_idx)

                self.create_horizontal_comparison(outputs, targets, batch_idx, val_dir)

                comparison = torch.cat([targets[0:1], outputs[0:1]], dim=3)
                grid = torchvision.utils.make_grid(comparison, nrow=1, normalize=True, scale_each=True)

        # Loss curves as requested
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(batch_indices, batch_nmse_losses, 'b-o')
        plt.title(f'NMSE per Batch')
        plt.xlabel('Batch Index')
        plt.ylabel('NMSE')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(batch_indices, batch_ssim_losses, 'r-o')
        plt.title(f'SSIM per Batch ')
        plt.xlabel('Batch Index')
        plt.ylabel('SSIM')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(batch_indices, batch_psnr_losses, 'g-o')
        plt.title(f'PSNR per Batch ')
        plt.xlabel('Batch Index')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(val_dir, "losses.png"))
        plt.close()

        avg_mse = total_mse / total_samples if total_samples > 0 else 0
        avg_rmse = math.sqrt(avg_mse)

        if total_energy == 0:
            avg_nmse = 0.0 if total_mse == 0 else float('inf')
        else:
            avg_nmse = total_mse / total_energy

        avg_ssim = total_ssim / total_samples if total_samples > 0 else 0
        avg_psnr = total_psnr / total_samples if total_samples > 0 else 0

        print(f"val NMSE: {avg_nmse:.4f}")
        print(f"val RMSE: {avg_rmse:.4f}")
        print(f"val SSIM: {avg_ssim:.4f}")
        print(f"val PSNR: {avg_psnr:.4f}")

        return avg_nmse, avg_rmse, avg_ssim, avg_psnr

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



