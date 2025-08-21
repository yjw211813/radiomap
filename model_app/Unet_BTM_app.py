import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
# from model.sub_block.loss_cal import DynamicLoss,FourierLoss
from model.sub_block.optimizer_schedule import DynamicLRScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import torch
import os
import shutil
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision


class Unet_BTM_app():
    def __init__(self, start_epoch, log_dir, warmup_epochs, model_save_dir):
        self.start_epoch = start_epoch
        self.log_dir = log_dir
        self.warmup_epochs = warmup_epochs
        self.model_save_dir = model_save_dir

    def evaluate(self, model, val_loader, device, writer, epoch):
        model.eval()  # Set model to evaluation mode
        total_samples = 0
        total_mse = 0.0
        total_energy = 0.0  # 用于NMSE的分母计算（目标的总能量）
        total_ssim = 0.0
        total_psnr = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Evaluating", ncols=100, leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # 获取当前batch的样本数
                batch_size = inputs.size(0)
                total_samples += batch_size

                # 计算MSE（整个batch的平均）
                criterion = nn.MSELoss()
                mse_batch = criterion(outputs, targets)
                total_mse += mse_batch.item() * batch_size  # 累加总MSE（未平均）

                # 计算NMSE所需的分母（目标向量的能量）
                energy_batch = criterion(targets, torch.zeros_like(targets))
                total_energy += energy_batch.item() * batch_size  # 累加总能量

                # 计算SSIM（整个batch的平均）
                ssim_batch = ssim(outputs, targets)
                total_ssim += ssim_batch.item() * batch_size

                # 计算PSNR（整个batch的平均）
                psnr_batch = psnr(outputs, targets)
                total_psnr += psnr_batch.item() * batch_size

        # 计算整个验证集的平均指标
        avg_mse = total_mse / total_samples  # 整个验证集的平均MSE
        avg_rmse = math.sqrt(avg_mse)  # RMSE
        avg_nmse = total_mse / total_energy  # NMSE = 总MSE / 总能量
        avg_ssim = total_ssim / total_samples
        avg_psnr = total_psnr / total_samples

        print(f"val NMSE: {avg_nmse:.4f}")
        print(f"val RMSE: {avg_rmse:.4f}")
        print(f"val SSIM: {avg_ssim:.4f}")
        print(f"val PSNR: {avg_psnr:.4f}")

        # Write validation metrics to TensorBoard
        writer.add_scalar('Loss/val', avg_rmse, epoch)
        # 可选：记录其他指标
        writer.add_scalar('NMSE/val', avg_nmse, epoch)
        writer.add_scalar('SSIM/val', avg_ssim, epoch)
        writer.add_scalar('PSNR/val', avg_psnr, epoch)

    def train(self, model, train_loader, val_loader, total_epoch, device, save_interval=1):

        eval_interval = 4
        # 清空 log_dir 下的文件（如果存在）
        if self.start_epoch == 0 and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=self.log_dir)  # TensorBoard SummaryWriter

        optimizer = optim.Adam(
            params=model.parameters(),
            lr=1e-4,  # 学习率
            betas=(0.9, 0.999),  # 动量参数
            weight_decay=0,  # L2正则化
            amsgrad=False  # 不使用AMSGrad
        )

        scheduler = DynamicLRScheduler(
            optimizer,
            lr_min=1e-6,  # 最小学习率
            lr_max=1e-3,  # 最大学习率
            warmup_epochs=self.warmup_epochs,  # 前warmup_epochs个epoch学习率上升
            decay_epochs=total_epoch - self.warmup_epochs  # 后面epoch学习率下降
        )
        # 如果提供了检查点路径，加载优化器和调度器状态
        # 这一次就直接只加载模型了下一次就优化器和模型一起加载
        if self.start_epoch != 0:
            checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_epoch_{self.start_epoch}.pth")
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            # model.load_state_dict(checkpoint)
            print("加载历史数据成功")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # for _ in range(self.start_epoch):
            #     scheduler.step()

        # 初始化动态损失
        criterion = torch.nn.MSELoss()
        model.to(device)

        for epoch in range(self.start_epoch, total_epoch):
            model.train()  # Set model to training mode
            running_loss = 0.0

            # 创建tqdm进度条
            train_loader_with_progress = tqdm(
                train_loader,
                desc=f'Epoch {epoch + 1}/{total_epoch}',  # 进度条前缀
                leave=True,  # 进度条完成后保留显示
                dynamic_ncols=True  # 自动调整宽度
            )

            for inputs, targets in train_loader_with_progress:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item() / inputs.shape[0]

                # 更新进度条的显示信息
                train_loader_with_progress.set_postfix(
                    loss=f'{loss.item() / inputs.shape[0]:.4f}',  # 当前批次的损失
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 再更新学习率（在每个epoch结束时）
            scheduler.step()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{total_epoch}], Train Loss: {avg_loss:.4f}")

            # Write loss to TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)

            # Evaluate the model after each epoch
            if (epoch + 1) % eval_interval == 0:
                self.evaluate(model, val_loader, device, writer, epoch)

            # Save the model checkpoint every `save_interval` epochs
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(checkpoint, os.path.join(self.model_save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))
                print("已经存储权重" + f"checkpoint_epoch_{epoch + 1}.pth")

        # 训练结束
        writer.close()


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

    def test(self, model, load_epoch, test_loader, device, val_dir):
        if load_epoch != 0:
            checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_epoch_{load_epoch}.pth")
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            print(f"加载历史数据load_epoch:{load_epoch}成功")
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()  # Set model to evaluation mode
        total_samples = 0
        total_mse = 0.0
        total_energy = 0.0  # 用于NMSE的分母计算（目标的总能量）
        total_ssim = 0.0
        total_psnr = 0.0
        # 创建目录保存验证结果图像
        os.makedirs(val_dir, exist_ok=True)
        # 用于跟踪每个batch的指标
        batch_nmse_losses = []
        batch_ssim_losses = []
        batch_psnr_losses = []
        batch_indices = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Testing", ncols=100, leave=False)):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # 获取当前batch的样本数
                batch_size = inputs.size(0)
                total_samples += batch_size

                # 计算MSE（整个batch的平均）
                criterion = nn.MSELoss()
                mse_batch = criterion(outputs, targets)
                total_mse += mse_batch.item() * batch_size  # 累加总MSE（未平均）

                # 计算NMSE所需的分母（目标向量的能量）
                energy_batch = criterion(targets, torch.zeros_like(targets))
                total_energy += energy_batch.item() * batch_size  # 累加总能量

                # 计算当前batch的NMSE
                if energy_batch.item() == 0:
                    nmse_loss_value = 0.0 if mse_batch.item() == 0 else float('inf')
                else:
                    nmse_loss_value = mse_batch.item() / energy_batch.item()

                # 计算SSIM（整个batch的平均）
                ssim_batch = ssim(outputs, targets)
                total_ssim += ssim_batch.item() * batch_size

                # 计算PSNR（整个batch的平均）
                psnr_batch = psnr(outputs, targets)
                total_psnr += psnr_batch.item() * batch_size

                # 记录当前batch的指标
                batch_nmse_losses.append(nmse_loss_value)
                batch_ssim_losses.append(ssim_batch.item())
                batch_psnr_losses.append(psnr_batch.item())
                batch_indices.append(batch_idx)

                # 调用create_horizontal_comparison函数
                self.create_horizontal_comparison(outputs, targets, batch_idx, val_dir)

                # ========== TensorBoard图像记录 ==========
                # 创建并排对比图
                comparison = torch.cat([targets[0:1], outputs[0:1]], dim=3)
                grid = torchvision.utils.make_grid(comparison, nrow=1, normalize=True, scale_each=True)

        # ========== 损失可视化 ==========
        plt.figure(figsize=(15, 5))

        # NMSE损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(batch_indices, batch_nmse_losses, 'b-o')
        plt.title(f'NMSE per Batch')
        plt.xlabel('Batch Index')
        plt.ylabel('NMSE')
        plt.grid(True)

        # SSIM曲线
        plt.subplot(1, 3, 2)
        plt.plot(batch_indices, batch_ssim_losses, 'r-o')
        plt.title(f'SSIM per Batch ')
        plt.xlabel('Batch Index')
        plt.ylabel('SSIM')
        plt.grid(True)

        # PSNR曲线
        plt.subplot(1, 3, 3)
        plt.plot(batch_indices, batch_psnr_losses, 'g-o')
        plt.title(f'PSNR per Batch ')
        plt.xlabel('Batch Index')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(val_dir, "losses.png"))
        plt.close()

        # 计算整个验证集的平均指标
        avg_mse = total_mse / total_samples  # 整个验证集的平均MSE
        avg_rmse = math.sqrt(avg_mse)  # RMSE

        # 处理总能量为0的情况
        if total_energy == 0:
            avg_nmse = 0.0 if total_mse == 0 else float('inf')
        else:
            avg_nmse = total_mse / total_energy  # NMSE = 总MSE / 总能量

        avg_ssim = total_ssim / total_samples
        avg_psnr = total_psnr / total_samples

        print(f"val NMSE: {avg_nmse:.4f}")
        print(f"val RMSE: {avg_rmse:.4f}")
        print(f"val SSIM: {avg_ssim:.4f}")
        print(f"val PSNR: {avg_psnr:.4f}")




