import os

import numpy as np
import shutil
import torch.optim as optim
from tqdm import tqdm

from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from model.flow_matching_model.Scheduler import GradualWarmupScheduler
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
import matplotlib.pyplot as plt
import math
import torch
from torch import nn


# Model wrapper class
class WrappedModel(ModelWrapper):
    def __init__(self, model, cfg_scale=7.0):
        super().__init__(model)
        self.cfg_scale = cfg_scale  # Classifier-Free Guidance scale

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        condition_info = extras.get("condition_info", None)
        # 如果不使用CFG，直接返回模型输出
        if self.cfg_scale == 0.0:
            return self.model(x, t, condition_info)
        # 使用CFG时，需要计算有条件和无条件的输出
        # 创建零条件信息
        zero_condition = torch.zeros_like(condition_info)
        # 计算无条件输出
        v_pred_uncond = self.model(x, t, zero_condition)
        # 计算有条件输出
        v_pred_cond = self.model(x, t, condition_info)
        # 应用CFG公式
        v_pred = v_pred_uncond + self.cfg_scale * (v_pred_cond - v_pred_uncond)
        return v_pred



class flowMatching_app():
    def __init__(self,start_epoch,model_save_dir,model_load_dir,writer,device,T):
        self.start_epoch = start_epoch
        self.board_writer = writer
        self.device = device
        self.model_save_dir = model_save_dir
        self.model_load_dir = model_load_dir
        self.T = T

    @staticmethod
    def visualize_time_series(sol, T_cpu, sample_idx, save_path):
        """
        可视化时间序列图像

        参数:
            sol: 解算结果，形状为 [时间步, 批次大小, 通道, 高度, 宽度]
            T_cpu: 时间点数组
            sample_idx: 要显示的样本索引
            save_path: 保存图像的路径
        """
        n_timesteps = len(T_cpu)
        n_cols = min(5, n_timesteps)  # 每行最多显示5个时间步
        n_rows = int(np.ceil(n_timesteps / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        fig.suptitle(f'Time Evolution of Sample {sample_idx}', fontsize=16)

        # 处理单行情况
        if n_rows == 1:
            axs = [axs] if n_cols == 1 else axs.reshape(1, -1)

        # 遍历所有时间步
        for i in range(n_timesteps):
            row_idx = i // n_cols
            col_idx = i % n_cols

            # 获取当前时间步的指定样本
            timestep_data = sol[i, sample_idx]
            single_image = timestep_data.squeeze()  # 移除通道维度

            # 显示图像
            ax = axs[row_idx][col_idx] if n_rows > 1 else axs[col_idx]
            im = ax.imshow(single_image, cmap='viridis', origin='lower')
            ax.set_title(f't = {T_cpu[i]:.2f}')
            ax.axis('off')

            # 添加颜色条
            fig.colorbar(im, ax=ax)

        # 隐藏多余的子图
        for i in range(n_timesteps, n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            ax = axs[row_idx][col_idx] if n_rows > 1 else axs[col_idx]
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 为标题留出空间

        # 保存时间序列图像
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved time series image to {save_path}")


    @staticmethod
    def visualize_target_image(targets, sample_idx, save_path):
        """
        可视化目标图像

        参数:
            targets: 目标图像数据
            sample_idx: 要显示的样本索引
            save_path: 保存图像的路径
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # 假设 targets 的形状为 (batch_size, 1, 32, 32)
        target_image = targets[sample_idx].squeeze().cpu().numpy()

        im = ax.imshow(target_image, cmap='viridis', origin='lower')
        ax.set_title(f'Target Image for Sample {sample_idx}')
        ax.axis('off')

        # 添加颜色条
        fig.colorbar(im, ax=ax)

        plt.tight_layout()

        # 保存目标图像
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved target image to {save_path}")


    def val(self,model, val_loader, val_dir):
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)  # 删除整个目录及其内容
        # 重新创建 log_dir
        os.makedirs(val_dir)

        model.eval()

        wrapped_vf = WrappedModel(model)
        solver = ODESolver(velocity_model=wrapped_vf)
        # 推理过程的步数 这个是总体推理相关设置
        T = torch.linspace(0, 1,  self.T // 10).to(self.device)
        # 两步之间的积分间隔 应该是ODE相关设置
        step_size = (T[1] - T[0]) / 10

        # 初始化指标累加器
        total_samples = 0
        total_mse = 0.0
        total_energy = 0.0  # 用于NMSE的分母计算（目标的总能量）
        total_ssim = 0.0
        total_psnr = 0.0

        # 加载模型并评估
        with torch.no_grad():
            # 遍历整个验证集
            for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc="Validating")):
                if batch_idx<3:
                    x_0 = inputs[:, 3, :, :].unsqueeze(1).to(self.device)# 将插值图像取出
                    condition_info = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # 使用ODE求解器获取解
                    sol = solver.sample(
                        time_grid=T,    # 时间网格 [t_0, t_1, ..., t_{N-1}]
                        x_init=x_0,     # 初始状态 x(t=0)
                        method='midpoint', # 使用中点法
                        step_size=step_size,# 积分步长
                        return_intermediates=True,# 返回所有时间点的解
                        condition_info=condition_info# 条件信息
                    )

                    # 获取最终时间步的解（t=1）
                    final_solution = sol[-1]  # 形状: [batch_size, ...]

                    # 计算当前批次的指标
                    batch_size = inputs.size(0)
                    total_samples += batch_size

                    # 计算MSE
                    mse_batch = nn.MSELoss()(final_solution, targets)
                    total_mse += mse_batch.item() * batch_size

                    # 计算NMSE所需的分母（目标向量的能量）
                    energy_batch = nn.MSELoss()(targets, torch.zeros_like(targets))
                    total_energy += energy_batch.item() * batch_size

                    # 计算SSIM（逐样本计算然后平均）
                    for i in range(batch_size):
                        ssim_val = ssim(final_solution[i], targets[i])
                        total_ssim += ssim_val

                    # 计算PSNR（逐样本计算然后平均）
                    for i in range(batch_size):
                        psnr_val = psnr(final_solution[i], targets[i])
                        total_psnr += psnr_val

                    # 只对第一个批次进行可视化
                    if batch_idx == 0:
                        sol_np = sol.cpu().numpy()
                        T_cpu = T.cpu().numpy()

                        # 选择要显示的样本索引
                        sample_idx = 0  # 可以更改为任何有效的索引值（0 到 batch_size-1）

                        # 1. 可视化时间序列图像
                        time_series_path = os.path.join(val_dir, "val_img.png")
                        self.visualize_time_series(sol_np, T_cpu, sample_idx, time_series_path)

                        # 2. 显示目标图像 (targets)
                        target_path = os.path.join(val_dir, "origin_img.png")
                        self.visualize_target_image(targets, sample_idx, target_path)

                        # 3. 显示所有图像
                        plt.show()

        # 计算整个验证集的平均指标
        avg_mse = total_mse / total_samples
        avg_rmse = math.sqrt(avg_mse)
        avg_nmse = total_mse / total_energy if total_energy != 0 else float('inf')
        avg_ssim = total_ssim / total_samples
        avg_psnr = total_psnr / total_samples

        print(f"Validation Results:")
        print(f"NMSE: {avg_nmse:.6f}")
        print(f"RMSE: {avg_rmse:.6f}")
        print(f"SSIM: {avg_ssim:.6f}")
        print(f"PSNR: {avg_psnr:.6f}")


    def train(self, model, train_loader, val_loader,val_dir, total_epoch, save_interval=1,val_interval=10,):

        # 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=total_epoch,
                                                               eta_min=0,
                                                               last_epoch=-1)
        warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer,
                                                 multiplier=1.5,               # 热身阶段最终会将学习率提高到初始值的2.5倍   1e-3 * multiplier
                                                 warm_epoch=total_epoch // 10, # 热身阶段占10%的总epoch
                                                 after_scheduler=cosineScheduler)
        # 定义采样轨迹 仿射概率路径 X_t = α_t * X_1 + σ_t * X_0 ， CondOTScheduler 是一个具体的调度器实现，它定义了线性插值路径
        path = AffineProbPath(scheduler=CondOTScheduler())

        # 使用MSELoss替代pow形式的损失计算
        criterion = torch.nn.MSELoss()

        # 如果提供了检查点路径，加载优化器和调度器状态
        if self.start_epoch != 0:
            checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_epoch_{self.start_epoch}.pth")
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            warmUpScheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"成功从epoch {self.start_epoch}恢复训练")

        model.to(self.device)

        for e in range(self.start_epoch, total_epoch):
            running_loss = 0.0

            with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
                for inputs, targets in tqdmDataLoader:
                    optimizer.zero_grad()

                    x_0 = inputs[:, 3, :, :].unsqueeze(1).to(self.device)# 将插值图像取出
                    x_1 = targets.to(self.device)
                    condition_info = inputs.to(self.device)
                    t = torch.rand(x_1.shape[0]).to(self.device)

                    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

                    if np.random.rand() < 0.1:
                        condition_info = torch.zeros_like(condition_info).to(self.device)

                    pred = model(path_sample.x_t, path_sample.t, condition_info)
                    loss = criterion(pred, path_sample.dx_t)
                    loss.backward()

                    # 这块还有一个超参数 下面是对梯度的向量长度进行截断
                    # 不改变梯度方向 但是将高于2 长度的向量全部变成向量长度为2 从而保证整体训练不会发散
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                    optimizer.step()
                    running_loss += loss.item()
                    # 更新一下进度条显示信息
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "img shape: ": x_1.shape,
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    })

            # 更新调度器
            warmUpScheduler.step()
            avg_loss = running_loss / len(tqdmDataLoader)
            self.board_writer.add_scalar('Loss/train', avg_loss, e)
            # 定期保存检查点，包括模型、优化器和调度器状态
            if (e + 1) % save_interval == 0:
                checkpoint = {
                    'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': warmUpScheduler.state_dict(),
                }
                torch.save(checkpoint, os.path.join(self.model_save_dir, f"checkpoint_epoch_{e + 1}.pth"))
                print(f"已保存检查点: checkpoint_epoch_{e + 1}.pth")
            if (e + 1) % val_interval == 0:
                self.val(self, model, val_loader, val_dir)
        self.board_writer.close()

    def test(self,model, load_epoch, test_loader, test_dir):
        # 重新创建 test_dir
        os.makedirs(test_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.model_load_dir, f"checkpoint_epoch_{load_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("model load weight done.")
        model.eval()

        wrapped_vf = WrappedModel(model)
        solver = ODESolver(velocity_model=wrapped_vf)
        # 推理过程的步数 这个是总体推理相关设置
        T = torch.linspace(0, 1,  self.T // 10).to(self.device)
        # 两步之间的积分间隔 应该是ODE相关设置
        step_size = (T[1] - T[0]) / 10

        # 初始化指标累加器
        total_samples = 0
        total_mse = 0.0
        total_energy = 0.0  # 用于NMSE的分母计算（目标的总能量）
        total_ssim = 0.0
        total_psnr = 0.0

        # 加载模型并评估
        with torch.no_grad():
            # 遍历整个验证集
            for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Validating")):
                if batch_idx<3:
                    x_0 = inputs[:, 3, :, :].unsqueeze(1).to(self.device)# 将插值图像取出
                    condition_info = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # 使用ODE求解器获取解
                    sol = solver.sample(
                        time_grid=T,    # 时间网格 [t_0, t_1, ..., t_{N-1}]
                        x_init=x_0,     # 初始状态 x(t=0)
                        method='midpoint', # 使用中点法
                        step_size=step_size,# 积分步长
                        return_intermediates=True,# 返回所有时间点的解
                        condition_info=condition_info# 条件信息
                    )

                    # 获取最终时间步的解（t=1）
                    final_solution = sol[-1]  # 形状: [batch_size, ...]

                    # 计算当前批次的指标
                    batch_size = inputs.size(0)
                    total_samples += batch_size

                    # 计算MSE
                    mse_batch = nn.MSELoss()(final_solution, targets)
                    total_mse += mse_batch.item() * batch_size

                    # 计算NMSE所需的分母（目标向量的能量）
                    energy_batch = nn.MSELoss()(targets, torch.zeros_like(targets))
                    total_energy += energy_batch.item() * batch_size

                    # 计算SSIM（逐样本计算然后平均）
                    for i in range(batch_size):
                        ssim_val = ssim(final_solution[i], targets[i])
                        total_ssim += ssim_val

                    # 计算PSNR（逐样本计算然后平均）
                    for i in range(batch_size):
                        psnr_val = psnr(final_solution[i], targets[i])
                        total_psnr += psnr_val

                    # 只对第一个批次进行可视化
                    if batch_idx == 0:
                        sol_np = sol.cpu().numpy()
                        T_cpu = T.cpu().numpy()

                        # 选择要显示的样本索引
                        sample_idx = 0  # 可以更改为任何有效的索引值（0 到 batch_size-1）

                        # 1. 可视化时间序列图像
                        time_series_path = os.path.join(test_dir, "val_img.png")
                        self.visualize_time_series(sol_np, T_cpu, sample_idx, time_series_path)

                        # 2. 显示目标图像 (targets)
                        target_path = os.path.join(test_dir, "origin_img.png")
                        self.visualize_target_image(targets, sample_idx, target_path)

                        # 3. 显示所有图像
                        plt.show()

        # 计算整个验证集的平均指标
        avg_mse = total_mse / total_samples
        avg_rmse = math.sqrt(avg_mse)
        avg_nmse = total_mse / total_energy if total_energy != 0 else float('inf')
        avg_ssim = total_ssim / total_samples
        avg_psnr = total_psnr / total_samples

        print(f"Validation Results:")
        print(f"NMSE: {avg_nmse:.6f}")
        print(f"RMSE: {avg_rmse:.6f}")
        print(f"SSIM: {avg_ssim:.6f}")
        print(f"PSNR: {avg_psnr:.6f}")


