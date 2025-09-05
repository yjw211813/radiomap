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

    def print_metrics(self, metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        print("{}: {}".format(phase, ", ".join(outputs)))

    def evaluate(self, model, val_loader, WNetPhase="firstU", targetType="dense", num_samples=300):
        model.eval()
        metrics = defaultdict(float)
        epoch_samples = 0

        with torch.no_grad():
            if targetType == "dense":
                for inputs, targets in tqdm(val_loader, desc="Evaluating", ncols=100, leave=False):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    [outputs1, outputs2] = model(inputs)
                    if WNetPhase == "firstU":
                        loss = self.calc_loss_dense(outputs1, targets, metrics)
                    else:
                        loss = self.calc_loss_dense(outputs2, targets, metrics)

                    epoch_samples += inputs.size(0)
            elif targetType == "sparse":
                for inputs, targets, samples in tqdm(val_loader, desc="Evaluating", ncols=100, leave=False):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    samples = samples.to(self.device)

                    [outputs1, outputs2] = model(inputs)
                    if WNetPhase == "firstU":
                        loss = self.calc_loss_sparse(outputs1, targets, samples, metrics, num_samples)
                    else:
                        loss = self.calc_loss_sparse(outputs2, targets, samples, metrics, num_samples)

                    epoch_samples += inputs.size(0)

        # 计算评估指标
        avg_loss = metrics['loss'] / epoch_samples

        # 计算额外的评估指标 (SSIM, PSNR)
        # 这里简化处理，实际可能需要更详细的指标计算
        if targetType == "dense":
            # 随机选择一个批次计算SSIM和PSNR
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                [outputs1, outputs2] = model(inputs)
                if WNetPhase == "firstU":
                    pred = outputs1
                else:
                    pred = outputs2

                # 计算SSIM和PSNR
                ssim_val = ssim(pred, targets)
                psnr_val = psnr(pred, targets)
                break

        print(f"Val Loss: {avg_loss:.4f}")
        if targetType == "dense":
            print(f"Val SSIM: {ssim_val:.4f}")
            print(f"Val PSNR: {psnr_val:.4f}")

        return avg_loss

    def train(self, model, train_loader, val_loader, num_epochs=50, WNetPhase="firstU",
              targetType="dense", num_samples=300, save_interval=10):

        # 初始化优化器和学习率调度器
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # 加载检查点（如果存在）
        if self.start_epoch != 0:
            checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_{WNetPhase}_epoch_{self.start_epoch}.pth")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint from epoch {self.start_epoch}")

        best_loss = 1e10
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(self.start_epoch, num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # 训练阶段
            model.train()
            metrics = defaultdict(float)
            epoch_samples = 0

            train_loader_with_progress = tqdm(
                train_loader,
                desc=f'Epoch {epoch + 1}/{num_epochs}',
                leave=True,
                dynamic_ncols=True
            )

            if targetType == "dense":
                for inputs, targets in train_loader_with_progress:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        [outputs1, outputs2] = model(inputs)
                        if WNetPhase == "firstU":
                            loss = self.calc_loss_dense(outputs1, targets, metrics)
                        else:
                            loss = self.calc_loss_dense(outputs2, targets, metrics)

                        loss.backward()
                        optimizer.step()

                    epoch_samples += inputs.size(0)
                    train_loader_with_progress.set_postfix(loss=f'{loss.item():.4f}')

            elif targetType == "sparse":
                for inputs, targets, samples in train_loader_with_progress:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    samples = samples.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        [outputs1, outputs2] = model(inputs)
                        if WNetPhase == "firstU":
                            loss = self.calc_loss_sparse(outputs1, targets, samples, metrics, num_samples)
                        else:
                            loss = self.calc_loss_sparse(outputs2, targets, samples, metrics, num_samples)

                        loss.backward()
                        optimizer.step()

                    epoch_samples += inputs.size(0)
                    train_loader_with_progress.set_postfix(loss=f'{loss.item():.4f}')

            # 更新学习率
            scheduler.step()

            # 打印训练指标
            self.print_metrics(metrics, epoch_samples, 'train')
            avg_train_loss = metrics['loss'] / epoch_samples
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)

            # 验证阶段
            avg_val_loss = self.evaluate(model, val_loader, WNetPhase, targetType, num_samples)
            self.writer.add_scalar('Loss/val', avg_val_loss, epoch)

            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print("Saving best model")

            # 定期保存检查点
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss
                }
                torch.save(checkpoint,
                           os.path.join(self.model_save_dir, f"checkpoint_{WNetPhase}_epoch_{epoch + 1}.pth"))
                print(f"Saved checkpoint at epoch {epoch + 1}")

        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)

        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(self.model_save_dir, f"Trained_Model_{WNetPhase}.pt"))
        print(f"Training completed. Best val loss: {best_loss:.4f}")

        return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 设置路径
    log_dir = r'/home/code/radio_map_construction/runs/model_log/model_compare/radioUnet/'
    model_save_dir = "/home/code/radio_map_construction/runs/model_compare/radioUnet/"

    # 创建训练器实例
    trainer = RadioWNetTrainer(start_epoch=0, log_dir=log_dir, model_save_dir=model_save_dir, device=device)

    # 读取数据
    Radio_train = loaders.RadioUNet_c(phase="train")
    Radio_val = loaders.RadioUNet_c(phase="val")

    batch_size = 15
    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1,
                            generator=torch.Generator(device=device)),
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1,
                          generator=torch.Generator(device=device))
    }

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.enabled

    # 训练第一个U
    model = modules.RadioWNet(phase="firstU")
    model.to(device)
    summary(model, input_size=(2, 256, 256))

    model = trainer.train(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=50,
        WNetPhase="firstU",
        targetType="dense"
    )

    # 训练第二个U
    model = modules.RadioWNet(phase="secondU")
    model.load_state_dict(torch.load(os.path.join(model_save_dir, "Trained_Model_FirstU.pt")))
    model.to(device)

    trainer.start_epoch = 0  # 重置起始epoch
    model = trainer.train(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=50,
        WNetPhase="secondU",
        targetType="dense"
    )