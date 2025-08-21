from model.sigle_Unet.Unet_BTM import Unet_BTM
# from model.metric_fun import NMSE
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from model.sub_block.loss_cal import DynamicLoss,FourierLoss
from model.sub_block.optimizer_schedule import DynamicLRScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import torch
from torch.utils.data import DataLoader
import os
import shutil
import math
from data.lib.loaders import RadioMapSeerLoader

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


simuSetDict = {
    "ind1": 0,  # 起始索引
    "ind2": 0,  # 末尾索引
    "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",  # 数据集文件夹
    "numTx": 80,  # 信源数量设定
    "thresh": 0.05,  # 环境噪声
    "simulation": "rand",  # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
    "carsSimul": "yes",  # 是否开启小车作为仿真
    "carsInput": "yes",  # 是否将小车图作为模型输入
    "IRT2maxW": 0.3,  # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
    "cityMap": "complete",  # 是否输入完全的城市地图
    "missing": 1,  # 地图缺失号码
    "fix_samples": 300,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
    "num_samples_low": 10,  # 最低采样数
    "num_samples_high": 300,  # 最高采样数
    "inter_flag":True # 看是否需要插值图像
}

train_batch_size = 32  # 批次大小
test_batch_size = 32  # 批次大小
warmup_epochs = 2
totally_epochs = 80
if simuSetDict["inter_flag"] == True:
    BTM_ghost_UNet_input_shape = [5, 256, 256]
else:
    BTM_ghost_UNet_input_shape = [4, 256, 256]
BTM_ghost_UNet_output_shape = [1, 256, 256]
C_down_list =  [32, 64, 128, 256]
C_list_attn = torch.tensor([64, 64, 64, 128, 128, 128, 128])
attn_params = [C_list_attn * 2, C_list_attn , C_list_attn // 2, C_list_attn // 2]
log_dir = r'/home/code/radio_map_construction/runs/model_log/UNet_SK'# log 存储位置
model_load_dir = "/home/code/radio_map_construction/runs/model_pth/UNet_SK/"# 模型加载目录
model_save_dir = "/home/code/radio_map_construction/runs/model_pth/UNet_SK/"# 模型存储位置
start_epoch = 15
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def evaluate(model, val_loader, device, writer, epoch):
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



def train(model, train_loader, val_loader, num_epochs, device, save_interval=1):


    global start_epoch
    global log_dir
    global model_save_dir

    eval_interval = 4
    # 清空 log_dir 下的文件（如果存在）
    if start_epoch == 0 and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard SummaryWriter

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=1e-4,                   # 学习率
        betas=(0.9, 0.999),         # 动量参数
        weight_decay=0,          # L2正则化
        amsgrad=False               # 不使用AMSGrad
    )

    scheduler = DynamicLRScheduler(
        optimizer,
        lr_min=1e-6,  # 最小学习率
        lr_max=1e-3,  # 最大学习率
        warmup_epochs=warmup_epochs,  # 前warmup_epochs个epoch学习率上升
        decay_epochs=num_epochs - warmup_epochs  # 后面epoch学习率下降
    )
    # 如果提供了检查点路径，加载优化器和调度器状态
    # 这一次就直接只加载模型了下一次就优化器和模型一起加载
    if start_epoch != 0:
        checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{start_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
        print("加载历史数据成功")
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for _ in range(start_epoch):
            scheduler.step()



    # 初始化动态损失
    criterion = torch.nn.MSELoss()
    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # 创建tqdm进度条
        train_loader_with_progress = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{num_epochs}',  # 进度条前缀
            leave=True,  # 进度条完成后保留显示
            dynamic_ncols=True  # 自动调整宽度
        )

        for inputs, targets in train_loader_with_progress:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()/inputs.shape[0]

            # 更新进度条的显示信息
            train_loader_with_progress.set_postfix(
                loss=f'{loss.item()/inputs.shape[0]:.4f}',  # 当前批次的损失
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 再更新学习率（在每个epoch结束时）
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        # Write loss to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        # Evaluate the model after each epoch
        if (epoch+1) % eval_interval == 0:
            evaluate(model, val_loader, device, writer, epoch)
        # Save the model checkpoint every `save_interval` epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(model_save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            print("已经存储权重"+f"checkpoint_epoch_{epoch+1}.pth")

    # 训练结束
    writer.close()


if __name__ == '__main__':
    print("inter_flag:",simuSetDict["inter_flag"])
    print("SK")
    Radio_train = RadioMapSeerLoader(simuSetDict, phase="train")
    Radio_val = RadioMapSeerLoader(simuSetDict, phase="val")
    Radio_test = RadioMapSeerLoader(simuSetDict, phase="test")

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=train_batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(Radio_val, batch_size=test_batch_size, shuffle=True, num_workers=4)
    }
    # 设置设备为GPU


    net = Unet_BTM(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    os.makedirs(model_save_dir, exist_ok=True)
    # 开始训练
    train(net, train_loader, val_loader, num_epochs=totally_epochs, device=device)

