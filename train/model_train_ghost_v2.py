from model.sigle_Unet.BTM_ghost_v2 import BTM_ghost_UNet_v2
# from model.metric_fun import NMSE
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import torch
from torch.utils.data import DataLoader
import os
import shutil
import math
from data.lib.loaders import RadioUNet_c_sprseIRT4

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

train_batch_size = 32  # 批次大小
test_batch_size = 32  # 批次大小
BTM_ghost_UNet_input_shape = [4, 256, 256]
BTM_ghost_UNet_output_shape = [1, 256, 256]
C_down_list =  [32, 64, 128, 256]
C_list_attn = torch.tensor([64, 64, 64, 128, 128, 128, 128])
attn_params = [C_list_attn * 2, C_list_attn , C_list_attn // 2, C_list_attn // 2]
log_dir = r'/home/code/radio_map_construction/runs/model_log/BTM_ghost_net_v2'
model_save_dir = "/home/code/radio_map_construction/runs/model_pth/BTM_ghost_net_v2/"
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def evaluate(model, val_loader, device, writer, epoch):
    model.eval()  # Set model to evaluation mode
    total_samples = 0
    total_mse = 0.0
    total_energy = 0.0  # 用于NMSE的分母计算（目标的总能量）
    total_ssim = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for inputs, targets, samples in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            samples = samples.to(device)
            samples = samples * targets
            inputs = torch.cat((inputs, samples), 1)

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


def train(model, train_loader, val_loader, num_epochs, device, save_interval=5):

    global log_dir
    global model_save_dir
    # 清空 log_dir 下的文件（如果存在）
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除整个目录及其内容
    # 重新创建 log_dir
    os.makedirs(log_dir)


    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard SummaryWriter
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    model.to(device)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, targets, samples in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            samples = samples.to(device)
            samples = samples * targets

            inputs = torch.cat((inputs, samples), 1)

            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()/inputs.shape[0]
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        # Write loss to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        # Evaluate the model after each epoch
        evaluate(model, val_loader, device, writer, epoch)
        # Save the model checkpoint every `save_interval` epochs
        if (epoch + 1) % save_interval == 0:
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    writer.close()


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


    net = BTM_ghost_UNet_v2(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)
    net.load_weights(os.path.join(model_save_dir, f"checkpoint_epoch_70.pth"))
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # 开始训练
    train(net, train_loader, val_loader, num_epochs=2000, device=device, save_interval=5)


  # import psutil
    # process = psutil.Process()
    # #设置CPU限制
    # process.nice(psutil.IDLE_PRIORITY_CLASS)
    # torch.set_num_threads(1)