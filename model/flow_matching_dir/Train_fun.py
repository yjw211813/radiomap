import os
from typing import Dict
import numpy as np
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

# from model.flow_matching_dir.flow_sample import GaussianDiffusionSampler, GaussianDiffusionTrainer
from model.flow_matching_dir.volecity_predict import velocity_UNet
from model.flow_matching_dir.Scheduler import GradualWarmupScheduler
from data.lib.loaders import RadioUNet_c_sprseIRT4
from torch.utils.tensorboard import SummaryWriter
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.distributions import Independent, Normal
import warnings
import time
import torch
from torch import nn, Tensor

# Model wrapper class
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        condition_info = extras["condition_info"]
        return self.model(x, t,condition_info)

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    C_down_list = modelConfig["C_down_list"]
    C_list_attn = torch.tensor(modelConfig["C_list_attn"])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]

    log_dir = modelConfig["log_dir"]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除整个目录及其内容
    # 重新创建 log_dir
    os.makedirs(log_dir)
    if os.path.exists(modelConfig["save_dir"]):
        shutil.rmtree(modelConfig["save_dir"])  # 删除整个目录及其内容
    # 重新创建 log_dir
    os.makedirs(modelConfig["save_dir"])
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard SummaryWriter

    # dataset
    Radio_train  = RadioUNet_c_sprseIRT4(phase="train", carsSimul="yes", carsInput="yes")
    dataloader = DataLoader(Radio_train, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = velocity_UNet(T = modelConfig["T"],
                           input_shape = modelConfig["UNet_input_shape"],
                           output_shape = modelConfig["UNet_output_shape"],
                           C_down_list = C_down_list,
                           attn_params = attn_params).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"]),
                                             map_location=device), strict=False)

        print("Model weight load down.")

    optimizer = torch.optim.Adam(net_model.parameters(), lr=modelConfig["lr"])
    # optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    # eta_min（学习率下限）
    # 控制调度器从哪个训练轮数开始计数，默认 last_epoch=-1 表示从 0 开始（即从头训练）
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    path = AffineProbPath(scheduler=CondOTScheduler())

    # trainer = GaussianDiffusionTrainer(
    #     net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    running_loss = 0.0
    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for condition_info, targets, samples in tqdmDataLoader:
                optimizer.zero_grad()
                b = condition_info.shape[0]
                # train
                condition_info = condition_info.to(device)
                x_1 = targets.to(device)
                samples = samples.to(device)
                samples = samples * x_1
                condition_info = torch.cat((condition_info, samples), 1)

                t = torch.rand(x_1.shape[0]).to(device)

                path_sample = path.sample(t=t, x_0=samples, x_1=x_1)


                if np.random.rand() < 0.1:
                    condition_info = torch.zeros_like(condition_info).to(device)
                loss = torch.pow(net_model(path_sample.x_t, path_sample.t,condition_info) - path_sample.dx_t, 2).mean()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                running_loss += loss.item()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_1.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        avg_loss = running_loss / len(tqdmDataLoader)
        writer.add_scalar('Loss/train', avg_loss, e)
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))




def eval(modelConfig: Dict):

    device = torch.device(modelConfig["device"])
    C_down_list = modelConfig["C_down_list"]
    C_list_attn = torch.tensor(modelConfig["C_list_attn"])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    Radio_test = RadioUNet_c_sprseIRT4(phase="test", carsSimul="yes", carsInput="yes")
    dataloader = DataLoader(Radio_test, batch_size=1, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    net_model = velocity_UNet(T=modelConfig["T"],
                           input_shape=modelConfig["UNet_input_shape"],
                           output_shape=modelConfig["UNet_output_shape"],
                           C_down_list=C_down_list,
                           attn_params=attn_params).to(device)

    ckpt = torch.load(os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)

    net_model.load_state_dict(ckpt)
    print("model load weight done.")
    net_model.eval()

    wrapped_vf = WrappedModel(net_model)
    solver = ODESolver(velocity_model=wrapped_vf)
    T = torch.linspace(0, 1, modelConfig["T"]//10).to(device)
    step_size = (T[1] - T[0])/10
    if os.path.exists(modelConfig["sampled_dir"]):
        shutil.rmtree(modelConfig["sampled_dir"])  # 删除整个目录及其内容
    # 重新创建 log_dir
    os.makedirs(modelConfig["sampled_dir"])

    # load model and evaluate
    with (torch.no_grad()):
        condition_info, targets, samples =  next(iter(dataloader))
        b = condition_info.shape[0]
        condition_info = condition_info.to(device)
        samples = samples.to(device)
        x_0 = targets.to(device)
        samples = samples * x_0
        condition_info = torch.cat((condition_info, samples), 1)

        sol = solver.sample(time_grid=T, x_init=samples, method='midpoint',
                            step_size=step_size, return_intermediates=True,condition_info = condition_info).cpu().numpy()
        T_cpu = T.cpu().numpy()

        # 选择要显示的样本索引
        sample_idx = 0  # 可以更改为任何有效的索引值（0 到 batch_size-1）

        # 1. 可视化时间序列图像
        # ========================================
        n_timesteps = len(T)
        n_cols = min(5, n_timesteps)  # 每行最多显示5个时间步
        n_rows = int(np.ceil(n_timesteps / n_cols))

        fig1, axs = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        fig1.suptitle(f'Time Evolution of Sample {sample_idx}', fontsize=16)

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
            fig1.colorbar(im, ax=ax)

        # 隐藏多余的子图
        for i in range(n_timesteps, n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            ax = axs[row_idx][col_idx] if n_rows > 1 else axs[col_idx]
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 为标题留出空间

        # 保存时间序列图像
        time_series_path = os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"])
        plt.savefig(time_series_path, bbox_inches='tight')
        print(f"Saved time series image to {time_series_path}")

        # 2. 显示目标图像 (targets)
        # ========================================
        fig2, ax = plt.subplots(figsize=(8, 8))

        # 假设 targets 的形状为 (batch_size, 1, 32, 32)
        target_image = targets[sample_idx].squeeze().cpu().numpy()

        im = ax.imshow(target_image, cmap='viridis', origin='lower')
        ax.set_title(f'Target Image for Sample {sample_idx}')
        ax.axis('off')

        # 添加颜色条
        fig2.colorbar(im, ax=ax)

        plt.tight_layout()

        # 保存目标图像
        target_path = os.path.join(modelConfig["sampled_dir"],  modelConfig["originalImgName"])
        plt.savefig(target_path, bbox_inches='tight')
        print(f"Saved target image to {target_path}")

        # 3. 显示所有图像
        # ========================================
        plt.show()
        # sampler = GaussianDiffusionSampler(model = net_model,
        #                                     beta_1 = modelConfig["beta_1"],
        #                                     beta_T = modelConfig["beta_T"],
        #                                     T = modelConfig["T"],
        #                                     w=modelConfig["w"]).to(device)
        # Sampled from standard normal distribution

        # noisyImage = torch.randn(size=[modelConfig["batch_size"], 1, modelConfig["img_H"], modelConfig["img_W"]], device=device)
        #
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        #
        # sampledImgs = sampler(noisyImage, condition_info)
        # # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # print(sampledImgs)
        # save_image(sampledImgs, os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        # save_image(x_0, os.path.join(modelConfig["sampled_dir"],  modelConfig["originalImgName"]), nrow=modelConfig["nrow"])