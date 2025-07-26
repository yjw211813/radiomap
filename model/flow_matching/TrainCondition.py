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

from model.diffusion_model.diffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from model.diffusion_model.noisePreCondition import noise_UNet
from Scheduler import GradualWarmupScheduler
from data.lib.loaders import RadioUNet_c_sprseIRT4
from torch.utils.tensorboard import SummaryWriter


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
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard SummaryWriter



    # dataset
    Radio_train  = RadioUNet_c_sprseIRT4(phase="train", carsSimul="yes", carsInput="yes")
    dataloader = DataLoader(Radio_train, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = noise_UNet(T = modelConfig["T"],
                           input_shape = modelConfig["UNet_input_shape"],
                           output_shape = modelConfig["UNet_output_shape"],
                           C_down_list = C_down_list,
                           attn_params = attn_params).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"]),
                                             map_location=device), strict=False)

        print("Model weight load down.")


    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    # eta_min（学习率下限）
    # 控制调度器从哪个训练轮数开始计数，默认 last_epoch=-1 表示从 0 开始（即从头训练）
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)

    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    running_loss = 0.0
    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for condition_info, targets, samples in tqdmDataLoader:
                optimizer.zero_grad()
                b = condition_info.shape[0]
                # train
                condition_info = condition_info.to(device)

                samples = samples.to(device)
                samples = samples * targets
                condition_info = torch.cat((condition_info, samples), 1)
                x_0 = targets.to(device)

                if np.random.rand() < 0.1:
                    condition_info = torch.zeros_like(condition_info).to(device)
                loss = trainer(x_0, condition_info).sum() / b ** 2.
                running_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
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
    dataloader = DataLoader(Radio_test, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    # load model and evaluate
    with (torch.no_grad()):
        condition_info, targets, samples =  next(iter(dataloader))
        b = condition_info.shape[0]
        condition_info = condition_info.to(device)
        samples = samples.to(device)
        samples = samples * targets
        condition_info = torch.cat((condition_info, samples), 1)
        x_0 = targets.to(device)

        net_model = noise_UNet(T=modelConfig["T"],
                               input_shape=modelConfig["UNet_input_shape"],
                               output_shape=modelConfig["UNet_output_shape"],
                               C_down_list=C_down_list,
                               attn_params=attn_params).to(device)

        ckpt = torch.load(os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)

        net_model.load_state_dict(ckpt)
        print("model load weight done.")
        net_model.eval()
        sampler = GaussianDiffusionSampler(model = net_model,
                                            beta_1 = modelConfig["beta_1"],
                                            beta_T = modelConfig["beta_T"],
                                            T = modelConfig["T"],
                                            w=modelConfig["w"]).to(device)
        # Sampled from standard normal distribution

        noisyImage = torch.randn(size=[modelConfig["batch_size"], 1, modelConfig["img_H"], modelConfig["img_W"]], device=device)

        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])

        sampledImgs = sampler(noisyImage, condition_info)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print(sampledImgs)
        save_image(sampledImgs, os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        save_image(x_0, os.path.join(modelConfig["sampled_dir"],  modelConfig["originalImgName"]), nrow=modelConfig["nrow"])