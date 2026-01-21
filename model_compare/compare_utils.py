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
from model_train.data_config import get_cars_load, get_nocars_load
from model.sigle_Unet.SAUnet_NoSA import SAUnetNoSA
from model.sigle_Unet.SAUnet_v0 import SAUnet_old


def create_multi_model_comparison(targets, outputs_dict, batch_idx, compare_dir):
    """
    高级模型对比可视化：包含原始输出、残差图及样本级指标
    """
    # 数据转换与维度处理
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    targets = targets.squeeze(1)  # [B, H, W]

    num_samples = min(4, targets.shape[0])
    model_names = list(outputs_dict.keys())
    num_cols = len(model_names) + 1  # 模型数 + 1 (Target)

    # 准备处理后的输出字典
    processed_outputs = {}
    for name, out in outputs_dict.items():
        if torch.is_tensor(out):
            out = out.cpu().numpy()
        processed_outputs[name] = out.squeeze(1)

    # 计算全局色阶范围（用于预测图对齐）
    all_vals = [targets[:num_samples]] + [v[:num_samples] for v in processed_outputs.values()]
    global_min = min(np.min(d) for d in all_vals)
    global_max = max(np.max(d) for d in all_vals)

    # 创建画布：行数为 2*num_samples (一行预测图，一行残差图)
    fig, axes = plt.subplots(num_samples * 2, num_cols,
                             figsize=(4 * num_cols, 4 * num_samples * 2),
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.1})

    for i in range(num_samples):
        # --- 第一列：绘制 Target ---
        ax_target = axes[i * 2, 0]
        im_t = ax_target.imshow(targets[i], cmap='jet', vmin=global_min, vmax=global_max)
        ax_target.set_title(f"Sample {i}\nGround Truth", fontweight='bold')
        ax_target.axis('off')

        # Target 下方留白或放置统计信息
        axes[i * 2 + 1, 0].axis('off')
        axes[i * 2 + 1, 0].text(0.5, 0.5, "Absolute\nError Map",
                                ha='center', va='center', fontweight='bold')

        # --- 后续列：绘制各模型输出及残差 ---
        for j, name in enumerate(model_names):
            col = j + 1
            pred = processed_outputs[name][i]
            target_i = targets[i]

            # 1. 绘制预测图
            ax_pred = axes[i * 2, col]
            ax_pred.imshow(pred, cmap='jet', vmin=global_min, vmax=global_max)

            # 计算该样本的 PSNR (简单转换回 tensor)
            data_range = float(max(global_max - global_min, 1e-5))  # 增加保护
            sample_psnr = psnr(torch.tensor(pred), torch.tensor(target_i), data_range=data_range)


            title_color = 'red' if name == 'SAUnet' else 'black'
            display_name = f"{name}(ours)" if name == 'SAUnet' else name
            ax_pred.set_title(f"{display_name}\nPSNR: {sample_psnr:.2f}dB", color=title_color)
            ax_pred.axis('off')

            # 2. 绘制残差图 (Error Map)
            ax_err = axes[i * 2 + 1, col]
            error_map = np.abs(pred - target_i)
            # 残差图使用不同的色阶，以突出显示误差
            im_err = ax_err.imshow(error_map, cmap='hot')
            ax_err.axis('off')

            # 为每个残差图添加一个小 colorbar
            plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

    # 保存图像
    save_path = os.path.join(compare_dir, f"batch_{batch_idx}_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_model_comparison_metrics(models, metrics, total_samples, compare_dir, figsize=(18, 12), dpi=300):
    """
    绘制多个模型的NMSE/SSIM/PSNR批次对比曲线和平均指标柱状图
    返回计算好的平均指标字典
    """
    # 参数检查
    if not isinstance(models, dict) or len(models) == 0:
        raise ValueError("models必须是非空字典")
    if not isinstance(metrics, dict) or len(metrics) == 0:
        raise ValueError("metrics必须是非空字典")
    if total_samples <= 0:
        raise ValueError("total_samples必须大于0")

    # 创建保存目录（如果不存在）
    os.makedirs(compare_dir, exist_ok=True)

    try:
        # 创建画布
        plt.figure(figsize=figsize)

        # 1. NMSE 批次对比（折线图）
        plt.subplot(2, 2, 1)
        for model_name in models.keys():
            # 检查当前模型的指标是否完整
            if model_name not in metrics:
                print(f"警告：模型{model_name}无对应的指标数据，跳过绘制")
                continue
            model_metrics = metrics[model_name]
            plt.plot(model_metrics['batch_indices'],
                     model_metrics['batch_nmse'],
                     'o-', label=model_name, markersize=3)
        plt.title('NMSE Comparison per Batch')
        plt.xlabel('Batch Index')
        plt.ylabel('NMSE')
        plt.legend()
        plt.grid(True)

        # 2. SSIM 批次对比（折线图）
        plt.subplot(2, 2, 2)
        for model_name in models.keys():
            if model_name not in metrics:
                continue
            model_metrics = metrics[model_name]
            plt.plot(model_metrics['batch_indices'],
                     model_metrics['batch_ssim'],
                     'o-', label=model_name, markersize=3)
        plt.title('SSIM Comparison per Batch')
        plt.xlabel('Batch Index')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(True)

        # 3. PSNR 批次对比（折线图）
        plt.subplot(2, 2, 3)
        for model_name in models.keys():
            if model_name not in metrics:
                continue
            model_metrics = metrics[model_name]
            plt.plot(model_metrics['batch_indices'],
                     model_metrics['batch_psnr'],
                     'o-', label=model_name, markersize=3)
        plt.title('PSNR Comparison per Batch')
        plt.xlabel('Batch Index')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid(True)

        # 计算各模型的平均指标
        avg_metrics = {}
        for model_name in models.keys():
            if model_name not in metrics:
                continue
            model_metrics = metrics[model_name]

            # 计算平均MSE和RMSE
            avg_mse = model_metrics['total_mse'] / total_samples
            avg_rmse = math.sqrt(avg_mse)

            # 计算平均NMSE（处理除零情况）
            if model_metrics['total_energy'] == 0:
                avg_nmse = 0.0 if model_metrics['total_mse'] == 0 else float('inf')
            else:
                avg_nmse = model_metrics['total_mse'] / model_metrics['total_energy']

            # 计算平均SSIM和PSNR
            avg_ssim = model_metrics['total_ssim'] / total_samples
            avg_psnr = model_metrics['total_psnr'] / total_samples

            avg_metrics[model_name] = {
                'NMSE': avg_nmse,
                'RMSE': avg_rmse,
                'SSIM': avg_ssim,
                'PSNR': avg_psnr
            }

        # 4. 平均指标对比（柱状图）
        plt.subplot(2, 2, 4)
        metric_names = ['NMSE', 'SSIM', 'PSNR']
        x = np.arange(len(metric_names))
        width = 0.25

        for i, model_name in enumerate(models.keys()):
            if model_name not in avg_metrics:
                continue
            values = [
                avg_metrics[model_name]['NMSE'],
                avg_metrics[model_name]['SSIM'],
                avg_metrics[model_name]['PSNR']
            ]
            # 对NMSE取对数以便更好地显示（避免log(0)）
            values[0] = np.log10(values[0] + 1e-10)
            plt.bar(x + i * width, values, width, label=model_name)

        plt.xlabel('Metrics')
        plt.ylabel('Values (log scale for NMSE)')
        plt.title('Average Metrics Comparison')
        plt.xticks(x + width, metric_names)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 调整布局并保存图片
        plt.tight_layout()
        save_path = os.path.join(compare_dir, "model_comparison_metrics.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"对比图已保存至：{save_path}")

        return avg_metrics  # 返回平均指标

    except Exception as e:
        raise RuntimeError(f"绘图过程中出错：{str(e)}")
    finally:
        # 确保关闭画布，释放资源
        plt.close()


def model_compare(models_dict, compare_dir, test_loader, device):
    """
    多个模型的对比分析 - 支持任意数量模型（通过字典传入）
    参数：
        models_dict: 模型字典，格式 {模型名: 模型实例}
        compare_dir: 对比结果保存目录
        test_loader: 测试数据加载器
        device: 计算设备 (cuda/cpu)
    """
    criterion = nn.MSELoss()

    # 为每个模型初始化指标存储
    metrics = {}
    for model_name in models_dict.keys():
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
        # 只测试前20个批次
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Model Comparison", ncols=100, leave=False)):
            if batch_idx >= 20:
                break

            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            # 获取各个模型的输出（核心改造：遍历模型字典）
            outputs_dict = {}
            for model_name, model in models_dict.items():
                if model_name == "SAUnet":
                    # SAUnet 专属处理逻辑
                    processed_inputs = preprocess_data(inputs, device)
                    model_outputs = model(processed_inputs)

                    # 【核心修复】：检查返回值是否为列表或元组
                    if isinstance(model_outputs, (list, tuple)):
                        model_outputs = model_outputs[0]

                    # 确保提取后再进行数值缩放
                    outputs_dict[model_name] = model_outputs * 256
                else:
                    # 其他模型通用处理逻辑
                    model_outputs = model(inputs)

                    # 【核心修复】：统一兼容列表或元组返回
                    if isinstance(model_outputs, (list, tuple)):
                        model_outputs = model_outputs[0]

                    outputs_dict[model_name] = model_outputs

            # 为每个模型计算指标
            for model_name, outputs in outputs_dict.items():

                mse_batch = criterion(outputs, targets)

                # 计算每个样本的能量，然后求和
                energy_per_sample = torch.mean(targets ** 2, dim=[1, 2, 3])  # 每个样本的平均能量
                energy_batch = torch.sum(energy_per_sample)  # 批次总能量

                metrics[model_name]['total_mse'] += mse_batch.item() * batch_size
                metrics[model_name]['total_energy'] += energy_batch.item()

                # 计算批次NMSE
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

    # 绘制指标图并获取平均指标
    avg_metrics = plot_model_comparison_metrics(models_dict, metrics, total_samples, compare_dir)

    # 打印详细指标
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS (First 20 batches)")
    print("=" * 60)

    for model_name in models_dict.keys():
        if model_name not in avg_metrics:
            continue
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


def get_radioUnet_model(base_dir, load_epoch, device):
    input_channels = 6
    WNetPhase = "secondU"
    radioUnet_model = RadioWNet(inputs=input_channels, phase=WNetPhase)
    radioUnet_model.to(device)
    radioUnet_model.eval()
    radioUnet_load_epoch = load_epoch
    radioUnet_save_dir = base_dir + r"/model_pth/RadioUnet/"  # 模型存储位置
    radioUnet_checkpoint_path = os.path.join(radioUnet_save_dir,
                                             f"checkpoint_{WNetPhase}_epoch_{radioUnet_load_epoch}.pth")
    radioUnet_checkpoint = torch.load(radioUnet_checkpoint_path, weights_only=True, map_location=device)

    radioUnet_model.load_state_dict(radioUnet_checkpoint['model_state_dict'])
    print(f"radioUnet加载历史数据load_epoch:{radioUnet_load_epoch}成功")
    return radioUnet_model


def get_UVM_model(base_dir, load_epoch, device):
    input_channels = 6
    UVM_model = UVMNet(n_channels=input_channels)
    UVM_model.to(device)
    UVM_model.eval()
    UVM_load_epoch = load_epoch
    UVM_save_dir = base_dir + r"/model_pth/UVM/"
    UVM_checkpoint_path = os.path.join(UVM_save_dir, f"checkpoint_epoch_{UVM_load_epoch}.pth")
    UVM_checkpoint = torch.load(UVM_checkpoint_path, weights_only=True, map_location=device)

    UVM_model.load_state_dict(UVM_checkpoint['model_state_dict'])
    print(f"UVM 加载历史数据load_epoch:{UVM_load_epoch}成功")
    return UVM_model


def get_REM_model(base_dir, load_epoch, device):
    input_channels = 6
    REMGAN_netG = modules.RadioWNet(inputs=input_channels, phase="firstU")
    REMGAN_netD = Discriminator()
    REMGAN_load_epoch = load_epoch
    REMGAN_netG.to(device)
    REMGAN_netD.to(device)
    REMGAN_save_dir = base_dir + r"/model_pth/REM_GAN/"
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
    return REMGAN_netG


def get_SAUNet_model(base_dir, load_epoch, device):
    input_shape = [6, 256, 256]
    output_shape = [1, 256, 256]
    C_down_list = [32, 64, 128, 256]
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    SAUNet_model = SAUnet_old(input_shape=input_shape, output_shape=output_shape, C_down_list=C_down_list,
                              attn_params=attn_params)
    load_epoch = load_epoch
    SAUNet_save_dir = base_dir + r"/model_pth/old_SAUnet/"

    SAUNet_checkpoint_path = os.path.join(SAUNet_save_dir, f"checkpoint_epoch_{load_epoch}.pth")
    SAUNet_checkpoint = torch.load(SAUNet_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{load_epoch}成功")
    SAUNet_model.load_state_dict(SAUNet_checkpoint['model_state_dict'])

    SAUNet_model.to(device)
    SAUNet_model.eval()  # Set model to evaluation mode
    return SAUNet_model


if __name__ == "__main__":
    # 设备初始化
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader, val_loader, test_loader = get_cars_load()

    # 路径配置
    base_dir = r"/home/code/radioMap/runs"
    compare_dir = base_dir + r"/model_val_log/compare/"
    sa_base_url = r"/home/code/radio_map_construction/runs"

    # 加载各个模型
    # UVM_model = get_UVM_model(base_dir, 20, device)
    radioUnet_model = get_radioUnet_model(base_dir, 95, device)

    SAUNet_model = get_SAUNet_model(sa_base_url, 34, device)
    # 可按需添加更多模型，例如：
    # REMGAN_model = get_REM_model(base_dir, 180, device)

    # 构建模型字典（核心：支持任意数量模型）
    models_dict = {
        'RadioUnet': radioUnet_model,
        # 'UVM': UVM_model,
        'SAUnet': SAUNet_model,
        # 'REMGAN': REMGAN_model  # 按需添加
    }

    # 执行模型对比（传入模型字典）
    avg_metrics = model_compare(models_dict, compare_dir, test_loader, device)