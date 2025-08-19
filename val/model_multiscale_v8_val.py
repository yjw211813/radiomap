import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

from model.sigle_Unet.Unet_BTM import BTM_multi_scale_v6
from data.lib.loaders import RadioUNet_c_sprseIRT4
from torch.utils.data import DataLoader
import shutil
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import math
train_batch_size = 32  # 批次大小
test_batch_size = 32  # 批次大小
BTM_ghost_UNet_input_shape = [4, 256, 256]
BTM_ghost_UNet_output_shape = [1, 256, 256]
C_down_list =  [32, 64, 128, 256]
C_list_attn = torch.tensor([64, 64, 64, 128, 128, 128, 128])
attn_params = [C_list_attn * 2, C_list_attn , C_list_attn // 2, C_list_attn // 2]
log_dir = r'/home/code/radio_map_construction/runs/model_val_log/BTM_multi_scale_v8_ssim'
model_save_dir = "/home/code/radio_map_construction/runs/model_pth/BTM_multi_scale_v9_ssim/"
pth_string = f"checkpoint_epoch_20.pth"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



def evaluate(model, val_loader, device, writer, epoch):
    model.eval()  # Set model to evaluation mode
    total_samples = 0
    total_mse = 0.0
    total_energy = 0.0  # 用于NMSE的分母计算（目标的总能量）
    total_ssim = 0.0
    total_psnr = 0.0

    # 创建目录保存验证结果图像
    os.makedirs("multiscale_v8_val_ressults", exist_ok=True)

    # 用于跟踪每个batch的指标
    batch_nmse_losses = []
    batch_ssim_losses = []
    batch_psnr_losses = []
    batch_indices = []


    with torch.no_grad():
        for batch_idx, (inputs, targets, samples) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            samples = samples.to(device)
            samples = samples * targets
            inputs = torch.cat((inputs, samples), 1)

            # 前向传播
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

            # ========== 可视化图像保存 ==========
            # 只取批次中的第一个样本进行可视化
            target_img = targets[0].cpu().numpy()
            output_img = outputs[0].cpu().numpy()

            # 处理单通道图像
            if target_img.shape[0] == 1:
                target_img = target_img.squeeze(0)
                output_img = output_img.squeeze(0)

            # 创建对比图像
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # 显示target
            ax = axes[0]
            im = ax.imshow(target_img, cmap='viridis')
            ax.set_title(f"Target (Batch {batch_idx})")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 显示output
            ax = axes[1]
            im = ax.imshow(output_img, cmap='viridis')
            ax.set_title(f"Output (Batch {batch_idx})")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(f"multiscale_v8_val_ressults/epoch_{epoch}_batch_{batch_idx}.png")
            plt.close()

            # ========== TensorBoard图像记录 ==========
            # 创建并排对比图
            comparison = torch.cat([targets[0:1], outputs[0:1]], dim=3)
            grid = torchvision.utils.make_grid(comparison, nrow=1, normalize=True, scale_each=True)
            writer.add_image(f'Validation/epoch_{epoch}_batch_{batch_idx}', grid, epoch)

    # ========== 损失可视化 ==========
    plt.figure(figsize=(15, 5))

    # NMSE损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(batch_indices, batch_nmse_losses, 'b-o')
    plt.title(f'NMSE per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('NMSE')
    plt.grid(True)

    # SSIM曲线
    plt.subplot(1, 3, 2)
    plt.plot(batch_indices, batch_ssim_losses, 'r-o')
    plt.title(f'SSIM per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('SSIM')
    plt.grid(True)

    # PSNR曲线
    plt.subplot(1, 3, 3)
    plt.plot(batch_indices, batch_psnr_losses, 'g-o')
    plt.title(f'PSNR per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"multiscale_v8_val_ressults/losses_epoch_{epoch}.png")
    plt.close()

    # 将损失曲线添加到TensorBoard
    loss_fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(batch_indices, batch_nmse_losses, 'b-o')
    plt.title(f'NMSE per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('NMSE')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(batch_indices, batch_ssim_losses, 'r-o')
    plt.title(f'SSIM per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('SSIM')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(batch_indices, batch_psnr_losses, 'g-o')
    plt.title(f'PSNR per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)

    plt.tight_layout()
    writer.add_figure('Validation/Losses', loss_fig, epoch)
    plt.close(loss_fig)

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

    # 将验证指标写入TensorBoard
    writer.add_scalar('Loss/val', avg_rmse, epoch)
    writer.add_scalar('NMSE/val', avg_nmse, epoch)
    writer.add_scalar('SSIM/val', avg_ssim, epoch)
    writer.add_scalar('PSNR/val', avg_psnr, epoch)

    return avg_rmse  # 可以作为主要验证指标返回



if __name__ == '__main__':

    Radio_train = RadioUNet_c_sprseIRT4(phase="train", carsSimul="yes", carsInput="yes")
    Radio_val = RadioUNet_c_sprseIRT4(phase="val", carsSimul="yes", carsInput="yes")
    Radio_test = RadioUNet_c_sprseIRT4(phase="test", carsSimul="yes", carsInput="yes")
    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }
    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=train_batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(Radio_val, batch_size=test_batch_size, shuffle=True, num_workers=4)
    }
    # 设置设备为GPU

    net = BTM_multi_scale_v6(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape, C_down_list, attn_params).to(device)

    net.load_weights(os.path.join(model_save_dir,pth_string ))
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # 清空 log_dir 下的文件（如果存在）
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除整个目录及其内容
    # 重新创建 log_dir
    os.makedirs(log_dir)


    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard SummaryWriter

    # 开始训练
    evaluate(net, val_loader, device, writer, epoch = 360)