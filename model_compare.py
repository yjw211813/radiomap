import torch
from model_app.SAUnet_app import SAUnet_app
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
import numpy as np
import pandas as pd
from model.sigle_Unet.simple_CNN import SAUnetForProcess
from model.sigle_Unet.SAUnet import SAUnet
from model_app.SAUnet_app import preprocess_data


def create_multi_model_comparison(targets, outputs_dict, batch_idx, compare_dir):
    """
    创建多个模型的对比图像 - 修改版本
    每个PNG只显示4个targets，targets放在最右边
    保证所有样本的热力图尺度和颜色映射保持一致
    """
    # 转换为numpy数组
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()

    # 移除通道维度
    targets = targets.squeeze(1)

    # 只取前4个样本
    num_samples = min(4, targets.shape[0])
    targets = targets[:num_samples]

    # 同样处理每个模型的输出，只取前4个
    processed_outputs_dict = {}
    for model_name, outputs in outputs_dict.items():
        if torch.is_tensor(outputs):
            outputs = outputs.cpu().numpy()
        outputs = outputs.squeeze(1)
        processed_outputs_dict[model_name] = outputs[:num_samples]

    num_models = len(processed_outputs_dict) + 1  # +1 for targets

    # 计算全局最小值和最大值，确保所有图像使用相同的颜色映射范围
    all_data = []
    for outputs in processed_outputs_dict.values():
        all_data.append(outputs)
    all_data.append(targets)

    # 计算全局范围
    global_min = min(np.min(data) for data in all_data)
    global_max = max(np.max(data) for data in all_data)

    # 创建一个大图像，包含所有模型和目标的对比
    # 布局：行数为样本数，列数为模型数，targets放在最右边
    fig, axes = plt.subplots(num_samples, num_models, figsize=(4 * num_models, 4 * num_samples))

    # 处理只有1个样本的情况
    if num_samples == 1:
        axes = axes.reshape(1, num_models)

    # 模型名称顺序（targets放在最后）
    model_names = list(processed_outputs_dict.keys()) + ['Target']

    for i in range(num_samples):
        # 先显示各个模型的输出
        for j, model_name in enumerate(processed_outputs_dict.keys()):
            output_img = processed_outputs_dict[model_name][i]

            ax = axes[i, j] if num_samples > 1 else axes[j]
            # 使用全局最小值和最大值确保颜色映射一致
            im = ax.imshow(output_img, cmap='jet', vmin=global_min, vmax=global_max)

            # 特殊处理BTMUNet标题
            if model_name == 'SAUnet':
                ax.set_title(f"SAUnet(ours)", color='red', fontweight='bold')
            else:
                ax.set_title(f"{model_name}")
            ax.axis('off')

        # 最后显示target（最右边）
        ax = axes[i, num_models - 1] if num_samples > 1 else axes[num_models - 1]
        # 使用相同的颜色映射范围
        im = ax.imshow(targets[i], cmap='jet', vmin=global_min, vmax=global_max)
        ax.set_title(f"Target")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f"batch_{batch_idx}_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def model_compare(radioUnet_model, UVM_model, REMGAN_netG, SAUnet_model, compare_dir, test_loader, device):
    """
    多个模型的对比分析 - 修改版本
    """
    criterion = nn.MSELoss()

    # 为每个模型初始化指标存储
    models = {
        'RadioUnet': radioUnet_model,
        'UVM': UVM_model,
        'REMGAN': REMGAN_netG,
        "SAUnet": SAUnet_model,
    }

    # 存储每个模型的指标
    metrics = {}
    for model_name in models.keys():
        metrics[model_name] = {
            'total_mse': 0.0,
            'total_energy': 0.0,
            'total_ssim': 0.0,
            'total_psnr': 0.0,
            'batch_nmse': [],
            'batch_ssim': [],
            'batch_psnr': [],
            'batch_indices': []
        }

    total_samples = 0
    os.makedirs(compare_dir, exist_ok=True)

    with torch.no_grad():
        # 只测试前10个批次
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Model Comparison", ncols=100, leave=False)):
            if batch_idx >= 20:  # 只测试前10个批次
                break

            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            # 获取各个模型的输出
            outputs_dict = {}

            # RadioUnet 输出
            radioUnet_outputs, _ = radioUnet_model(inputs)
            outputs_dict['RadioUnet'] = radioUnet_outputs

            # UVM 输出
            UVM_outputs = UVM_model(inputs)
            outputs_dict['UVM'] = UVM_outputs

            # REMGAN 输出
            REMGAN_outputs, _ = REMGAN_netG(inputs)
            outputs_dict['REMGAN'] = REMGAN_outputs

            # SAUnet 输出
            inputs = preprocess_data(inputs, device)
            SAUnet_outputs = SAUnet_model(inputs)

            outputs_dict['SAUnet'] = SAUnet_outputs * 256

            # 为每个模型计算指标
            for model_name, outputs in outputs_dict.items():
                # 计算损失和指标
                mse_batch = criterion(outputs, targets)

                # 修正：计算每个样本的能量，然后求和
                energy_per_sample = torch.mean(targets ** 2, dim=[1, 2, 3])  # 每个样本的平均能量
                energy_batch = torch.sum(energy_per_sample)  # 批次总能量

                metrics[model_name]['total_mse'] += mse_batch.item() * batch_size
                metrics[model_name]['total_energy'] += energy_batch.item()

                # 修正NMSE计算：使用批次内平均
                if energy_batch.item() == 0:
                    nmse_loss_value = 0.0 if mse_batch.item() == 0 else float('inf')
                else:
                    nmse_loss_value = (mse_batch.item() * batch_size) / energy_batch.item()

                ssim_batch = ssim(outputs, targets)
                metrics[model_name]['total_ssim'] += ssim_batch.item() * batch_size

                psnr_batch = psnr(outputs, targets)
                metrics[model_name]['total_psnr'] += psnr_batch.item() * batch_size

                metrics[model_name]['batch_nmse'].append(nmse_loss_value)
                metrics[model_name]['batch_ssim'].append(ssim_batch.item())
                metrics[model_name]['batch_psnr'].append(psnr_batch.item())
                metrics[model_name]['batch_indices'].append(batch_idx)

            # 创建多模型对比图
            create_multi_model_comparison(targets, outputs_dict, batch_idx, compare_dir)

    # 绘制对比曲线
    plt.figure(figsize=(18, 12))

    # NMSE 对比
    plt.subplot(2, 2, 1)
    for model_name in models.keys():
        plt.plot(metrics[model_name]['batch_indices'],
                 metrics[model_name]['batch_nmse'],
                 'o-', label=model_name, markersize=3)
    plt.title('NMSE Comparison per Batch')
    plt.xlabel('Batch Index')
    plt.ylabel('NMSE')
    plt.legend()
    plt.grid(True)

    # SSIM 对比
    plt.subplot(2, 2, 2)
    for model_name in models.keys():
        plt.plot(metrics[model_name]['batch_indices'],
                 metrics[model_name]['batch_ssim'],
                 'o-', label=model_name, markersize=3)
    plt.title('SSIM Comparison per Batch')
    plt.xlabel('Batch Index')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)

    # PSNR 对比
    plt.subplot(2, 2, 3)
    for model_name in models.keys():
        plt.plot(metrics[model_name]['batch_indices'],
                 metrics[model_name]['batch_psnr'],
                 'o-', label=model_name, markersize=3)
    plt.title('PSNR Comparison per Batch')
    plt.xlabel('Batch Index')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    # 计算并显示平均指标
    avg_metrics = {}
    for model_name in models.keys():
        avg_mse = metrics[model_name]['total_mse'] / total_samples
        avg_rmse = math.sqrt(avg_mse)

        # 修正avg_nmse计算：使用总MSE和总能量
        if metrics[model_name]['total_energy'] == 0:
            avg_nmse = 0.0 if metrics[model_name]['total_mse'] == 0 else float('inf')
        else:
            avg_nmse = metrics[model_name]['total_mse'] / metrics[model_name]['total_energy']

        avg_ssim = metrics[model_name]['total_ssim'] / total_samples
        avg_psnr = metrics[model_name]['total_psnr'] / total_samples

        avg_metrics[model_name] = {
            'NMSE': avg_nmse,
            'RMSE': avg_rmse,
            'SSIM': avg_ssim,
            'PSNR': avg_psnr
        }

    # 平均指标柱状图
    plt.subplot(2, 2, 4)
    metric_names = ['NMSE', 'SSIM', 'PSNR']
    x = np.arange(len(metric_names))
    width = 0.25

    for i, model_name in enumerate(models.keys()):
        values = [
            avg_metrics[model_name]['NMSE'],
            avg_metrics[model_name]['SSIM'],
            avg_metrics[model_name]['PSNR']
        ]
        # 对NMSE取对数以便更好地显示
        values[0] = np.log10(values[0] + 1e-10)  # 避免log(0)
        plt.bar(x + i * width, values, width, label=model_name)

    plt.xlabel('Metrics')
    plt.ylabel('Values (log scale for NMSE)')
    plt.title('Average Metrics Comparison')
    plt.xticks(x + width, metric_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "model_comparison_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 打印详细指标
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS (First 10 batches)")
    print("=" * 60)

    for model_name in models.keys():
        print(f"\n{model_name}:")
        print(f"  NMSE:  {avg_metrics[model_name]['NMSE']:.6f}")
        print(f"  RMSE:  {avg_metrics[model_name]['RMSE']:.6f}")
        print(f"  SSIM:  {avg_metrics[model_name]['SSIM']:.6f}")
        print(f"  PSNR:  {avg_metrics[model_name]['PSNR']:.6f} dB")

    # 保存指标到CSV文件
    metrics_df = pd.DataFrame(avg_metrics).T
    metrics_df.to_csv(os.path.join(compare_dir, "model_comparison_metrics.csv"))

    print(f"\nComparison results saved to: {compare_dir}")

    return avg_metrics


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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

    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 4
    # 加载数据集

    Radio_test = RadioMapSeerLoader(simuSetDict, phase="test")

    test_loader = DataLoader(Radio_test, batch_size=test_batch_size, shuffle=True, num_workers=4)

    compare_dir = r"/home/code/radio_map_construction/runs/model_val_log/compare/"

    input_channels = 6
    WNetPhase = "secondU"
    radioUnet_model = RadioWNet(inputs=input_channels, phase=WNetPhase)
    radioUnet_model.to(device)
    radioUnet_model.eval()
    radioUnet_load_epoch = 96
    radioUnet_save_dir = r"/home/code/radio_map_construction/runs/model_pth/RadioUnet/"  # 模型存储位置
    radioUnet_checkpoint_path = os.path.join(radioUnet_save_dir,
                                             f"checkpoint_{WNetPhase}_epoch_{radioUnet_load_epoch}.pth")
    radioUnet_checkpoint = torch.load(radioUnet_checkpoint_path, weights_only=True, map_location=device)

    radioUnet_model.load_state_dict(radioUnet_checkpoint['model_state_dict'])
    print(f"radioUnet加载历史数据load_epoch:{radioUnet_load_epoch}成功")

    UVM_model = UVMNet(n_channels=input_channels)
    UVM_model.to(device)
    UVM_model.eval()
    UVM_load_epoch = 20
    UVM_save_dir = r"/home/code/radio_map_construction/runs/model_pth/UVM/"
    UVM_checkpoint_path = os.path.join(UVM_save_dir, f"checkpoint_epoch_{UVM_load_epoch}.pth")
    UVM_checkpoint = torch.load(UVM_checkpoint_path, weights_only=True, map_location=device)

    UVM_model.load_state_dict(UVM_checkpoint['model_state_dict'])
    print(f"UVM 加载历史数据load_epoch:{UVM_load_epoch}成功")

    REMGAN_netG = modules.RadioWNet(inputs=input_channels, phase="firstU")
    REMGAN_netD = Discriminator()
    REMGAN_load_epoch = 150
    REMGAN_netG.to(device)
    REMGAN_netD.to(device)
    REMGAN_save_dir = r"/home/code/radio_map_construction/runs/model_pth/REM_GAN/"
    # 加载最佳检查点
    best_checkpoint_path = os.path.join(REMGAN_save_dir, f"checkpoint_REMGAN_epoch_{REMGAN_load_epoch}.pth")
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

    BTM_ghost_UNet_input_shape = [6, 256, 256]
    BTM_ghost_UNet_output_shape = [1, 256, 256]
    C_down_list = [64, 128, 256, 512]
    C_list_attn = torch.tensor([64, 64, 128, 128, 256])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    SAUNet_model = SAUnet(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params)
    load_epoch = 24
    SAUNet_save_dir = r"/home/code/radio_map_construction/runs/model_pth/old_SAUnet_deeper/"

    SAUNet_checkpoint_path = os.path.join(SAUNet_save_dir, f"checkpoint_epoch_{load_epoch}.pth")
    SAUNet_checkpoint = torch.load(SAUNet_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{load_epoch}成功")
    SAUNet_model.load_state_dict(SAUNet_checkpoint['model_state_dict'])

    SAUNet_model.to(device)
    SAUNet_model.eval()  # Set model to evaluation mode

    avg_metrics = model_compare(radioUnet_model, UVM_model, REMGAN_netG, SAUNet_model, compare_dir, test_loader,
                                device)