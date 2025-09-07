import torch
import torch.nn as nn
import copy
import time
from collections import defaultdict
import os
from data.lib import loaders
from model.radioUnetModel import modules
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import torchvision


class RadioWNetTrainer:
    def __init__(self, start_epoch, log_dir, model_save_dir, device):
        self.start_epoch = start_epoch
        self.log_dir = log_dir
        self.model_save_dir = model_save_dir
        self.device = device
        self.targetType = "dense"
        # 清空 log_dir 下的文件（如果存在）
        if self.start_epoch == 0 and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def calc_loss_dense(self, pred, target, metrics):
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        return loss

    def calc_loss_sparse(self, pred, target, samples, metrics, num_samples):
        criterion = nn.MSELoss()
        loss = criterion(samples * pred, samples * target) * (256 ** 2) / num_samples
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        return loss


    def val(self):

        print("val start")

    def train(self,model,train_loader,val_loader, total_epoch, WNetPhase="firstU", save_interval=1):
        eval_interval = 4
        # 初始化最佳权重
        best_model_wts = copy.deepcopy(model.state_dict())
        # 初始化最佳loss
        best_loss = 1e10
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        if self.start_epoch != 0:
            # 记住最佳的状态
            # 记住最佳的loss
            # 从最佳状态恢复
            self.start_epoch = best_epoch
            checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_epoch_best.pth")
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            # model.load_state_dict(checkpoint)
            print("加载历史数据成功")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for epoch in range(self.start_epoch, total_epoch):
            since = time.time()
            model.train()  # Set model to training mode
            # 创建tqdm进度条
            train_loader_with_progress = tqdm(
                train_loader,
                desc=f'Epoch {epoch + 1}/{total_epoch}',  # 进度条前缀
                leave=True,  # 进度条完成后保留显示
                dynamic_ncols=True  # 自动调整宽度
            )
            for inputs, targets in train_loader_with_progress:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)





            time_elapsed = time.time() - since
        print("train start")

    def test(self):

        print("test start")
