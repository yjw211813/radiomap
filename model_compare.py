import torch
from model_app.Unet_BTM_app import Unet_BTM_app
from data.lib.seer_loader import RadioMapSeerLoader
from torch.utils.data import DataLoader
from model.UVM.UVM_model import UVMNet
import os
from model.radioUnet.RadioUnetModel import RadioWNet
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torchvision
from model.rem_gan import modules
from model.rem_gan.EncoderModels import ResnetGenerator, Discriminator





def create_horizontal_comparison( outputs, targets, batch_idx, val_dir):
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


def model_compare(radioUnet_model,UVM_model,REMGAN_netG,compare_dir,test_loader,device):
# 多个模型拼在一块的对比图
    criterion = nn.MSELoss()
    total_samples = 0
    total_mse = 0.0
    total_energy = 0.0
    total_ssim = 0.0
    total_psnr = 0.0

    os.makedirs(compare_dir, exist_ok=True)

    batch_nmse_losses = []
    batch_ssim_losses = []
    batch_psnr_losses = []
    batch_indices = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing", ncols=100, leave=False)):

            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            [_, radioUnet_outputs] = radioUnet_model(inputs)

            UVM_outputs = UVM_model(inputs)

            REMGAN_outputs, _ = REMGAN_netG(inputs)



            batch_size = inputs.size(0)
            total_samples += batch_size

            # 计算损失和指标
            mse_batch = criterion(outputs, targets)
            energy_batch = criterion(targets, torch.zeros_like(targets))

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

            create_horizontal_comparison(outputs, targets, batch_idx, compare_dir)

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
    plt.savefig(os.path.join(compare_dir, "losses.png"))
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
# 多个模型的曲线图

    print(1)





if __name__ == "__main__":
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


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
        "fix_samples": 655,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
        "num_samples_low": 10,  # 最低采样数
        "num_samples_high": 300,  # 最高采样数
        "inter_flag": True,  # 看是否需要插值图像
        "scale256_flag": True,  # 取值范围是否为0 - 255
        "sample_flag": True,  # 是否有采样输入
        "loss_samples_flag": False,  # 是否定义loss为稀疏采样loss
        "formula_flag": True
    }

    train_batch_size = 16
    val_batch_size = 16
    test_batch_size = 16
    # 加载数据集

    Radio_test = RadioMapSeerLoader(simuSetDict, phase="test")

    test_loader = DataLoader(Radio_test, batch_size=test_batch_size, shuffle=True, num_workers=4)

    compare_dir = r"/home/code/radio_map_construction/runs/model_val_log/compare/"


    WNetPhase = "secondU"
    radioUnet_model = RadioWNet(inputs=2, phase=WNetPhase)
    radioUnet_model.to(device)
    radioUnet_model.eval()
    radioUnet_load_epoch = 0
    radioUnet_save_dir = r"/home/code/radio_map_construction/runs/model_pth/RadioUnet/"  # 模型存储位置
    radioUnet_checkpoint_path = os.path.join(radioUnet_save_dir, f"checkpoint_{WNetPhase}_epoch_{radioUnet_load_epoch}.pth")
    radioUnet_checkpoint = torch.load(radioUnet_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{radioUnet_load_epoch}成功")
    radioUnet_model.load_state_dict(radioUnet_checkpoint['model_state_dict'])


    UVM_model = UVMNet(n_channels = 6)
    UVM_model.to(device)
    UVM_model.eval()
    UVM_load_epoch = 1
    UVM_save_dir = r"/home/code/radio_map_construction/runs/model_pth/UVM/"
    UVM_checkpoint_path = os.path.join(UVM_save_dir, f"checkpoint_epoch_{UVM_load_epoch}.pth")
    UVM_checkpoint = torch.load(UVM_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{UVM_load_epoch}成功")
    UVM_model.load_state_dict(UVM_checkpoint['model_state_dict'])

    REMGAN_netG = modules.RadioWNet(inputs=6,phase="firstU")
    REMGAN_netD = Discriminator()
    REMGAN_netG.to(device)
    REMGAN_netD.to(device)
    REMGAN_save_dir =  r"/home/code/radio_map_construction/runs/model_pth/REM_GAN/"
    # 加载最佳检查点
    best_checkpoint_path = os.path.join(REMGAN_save_dir, "checkpoint_REMGAN_best.pth")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        REMGAN_netG.load_state_dict(checkpoint['netG_state_dict'])
        REMGAN_netD.load_state_dict(checkpoint['netD_state_dict'])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['best_loss']:.6f}")
    else:
        print("Warning: Best checkpoint not found. Using current model weights.")

    # 设置模型为评估模式
    REMGAN_netG.eval()
    REMGAN_netD.eval()



