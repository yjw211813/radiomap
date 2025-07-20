import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model.metric_fun import NMSE,SSIM,PSNR
from model.UNet_model import BTM_ghost_UNet
from data.lib.loaders import RadioUNet_c_sprseIRT4
from torch.utils.data import Dataset, DataLoader
import shutil
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, val_loader, device, writer, epoch):
    model.eval()  # Set model to evaluation mode
    nmse_loss = NMSE()
    ssim_loss = SSIM(L=1.0)
    psnr_loss = PSNR(r=1.0)

    # Create directory for saving images
    os.makedirs("val_results", exist_ok=True)

    # Lists to store batch losses for visualization
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

            # Forward pass
            outputs = model(inputs)

            # Calculate losses
            loss = torch.nn.MSELoss()(outputs, targets)
            nmse_loss_value = nmse_loss(outputs, targets)
            ssim_loss_value = ssim_loss(outputs, targets)
            psnr_loss_value = psnr_loss(outputs, targets)

            # Store batch losses
            batch_nmse_losses.append(nmse_loss_value.item())
            batch_ssim_losses.append(ssim_loss_value.item())
            batch_psnr_losses.append(psnr_loss_value.item())
            batch_indices.append(batch_idx)

            # ========== 可视化图像保存 ==========
            # 只取批次中的第一个样本进行可视化
            target_img = targets[0].cpu().numpy()
            output_img = outputs[0].cpu().numpy()

            # 处理多通道图像
            if target_img.shape[0] == 1:  # 单通道图像
                target_img = target_img.squeeze(0)
                output_img = output_img.squeeze(0)
            else:  # 多通道图像 (C, H, W) -> (H, W, C)
                target_img = np.moveaxis(target_img, 0, -1)
                output_img = np.moveaxis(output_img, 0, -1)

            # 创建对比图像
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # 显示target
            ax = axes[0]
            ax.imshow(target_img, cmap='viridis' if len(target_img.shape) == 2 else None)
            ax.set_title(f"Target (Batch {batch_idx})")
            ax.axis('off')

            # 显示output
            ax = axes[1]
            ax.imshow(output_img, cmap='viridis' if len(output_img.shape) == 2 else None)
            ax.set_title(f"Output (Batch {batch_idx})")
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(f"val_results/epoch_{epoch}_batch_{batch_idx}.png")
            plt.close()

            # ========== TensorBoard图像记录 ==========
            # 创建并排对比图
            comparison = torch.cat([targets[0].unsqueeze(0), outputs[0].unsqueeze(0)], dim=3)
            grid = torchvision.utils.make_grid(comparison, nrow=1, normalize=True, scale_each=True)
            writer.add_image(f'Validation/epoch_{epoch}_batch_{batch_idx}', grid, epoch)

    # ========== 损失可视化 ==========
    plt.figure(figsize=(15, 5))

    # NMSE损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(batch_indices, batch_nmse_losses, 'b-o')
    plt.title(f'NMSE Loss per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('NMSE Loss')
    plt.grid(True)

    # SSIM损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(batch_indices, batch_ssim_losses, 'r-o')
    plt.title(f'SSIM Loss per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('SSIM Loss')
    plt.grid(True)

    # PSNR损失曲线
    plt.subplot(1, 3, 3)
    plt.plot(batch_indices, batch_psnr_losses, 'g-o')
    plt.title(f'PSNR Loss per Batch (Epoch {epoch})')
    plt.xlabel('Batch Index')
    plt.ylabel('PSNR Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"val_results/losses_epoch_{epoch}.png")
    plt.close()

    # 将损失曲线添加到TensorBoard
    loss_fig = plt.figure(figsize=(15, 5))
    # ... [同上创建三个子图] ...
    writer.add_figure('Validation/Losses', loss_fig, epoch)



if __name__ == '__main__':
    # import psutil
    # process = psutil.Process()
    # #设置CPU限制
    # process.nice(psutil.IDLE_PRIORITY_CLASS)
    # torch.set_num_threads(1)

    Radio_train = RadioUNet_c_sprseIRT4(phase="train", carsSimul="yes", carsInput="yes")
    Radio_val = RadioUNet_c_sprseIRT4(phase="val", carsSimul="yes", carsInput="yes")
    Radio_test = RadioUNet_c_sprseIRT4(phase="test", carsSimul="yes", carsInput="yes")
    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    train_batch_size = 8  # 批次大小
    test_batch_size = 8  # 批次大小

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=train_batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(Radio_val, batch_size=test_batch_size, shuffle=True, num_workers=4)
    }

    # 设置设备为GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cpu')

    BTM_ghost_UNet_input_shape = [4, 256, 256]
    BTM_ghost_UNet_output_shape = [1, 256, 256]
    C_down_list = [64, 128, 256, 512]
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    net = BTM_ghost_UNet(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,C_list_attn).to(device)
    net.load_weights("../runs/model_pth/BTM_ghost_net/checkpoint_epoch_200.pth")
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    log_dir = r'../runs/model_log/BTM_ghost_net_val'
    # 清空 log_dir 下的文件（如果存在）
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除整个目录及其内容
    # 重新创建 log_dir
    os.makedirs(log_dir)


    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard SummaryWriter

    # 开始训练
    evaluate(net, val_loader, device, writer, epoch = 100)