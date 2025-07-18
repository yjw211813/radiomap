from model.UNet_model import BTM_ghost_UNet
from model.metric_fun import NMSE,SSIM,PSNR
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import sys
import shutil
import h5py
from data.lib.loaders import RadioUNet_c_sprseIRT4

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter





# 假设 topo_classify_net 已经定义好了
# my_net = topo_classify_net(emb_dim=64, shape_dict=shape_dict, class_num=100)

def evaluate(model, val_loader, device, writer, epoch):
    model.eval()  # Set model to evaluation mode
    nmse_loss = NMSE()
    ssim_loss = SSIM(L=1.0)
    psnr_loss = PSNR(r=1.0)
    running_loss = 0.0
    running_nmse_loss = 0.0
    running_ssim_loss = 0.0
    running_psnr_loss = 0.0
    with torch.no_grad():
        for inputs, targets, samples in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            samples = samples.to(device)
            samples = samples * targets

            inputs = torch.cat((inputs, samples), 1)

            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss = torch.nn.MSELoss()(outputs, targets)
            running_loss += loss.item()/inputs.shape[0]
            nmse_loss_value = nmse_loss(outputs, targets)
            running_nmse_loss+= nmse_loss_value.item()
            ssim_loss_value = ssim_loss(outputs, targets)
            running_ssim_loss += ssim_loss_value.item()
            psnr_loss_value = psnr_loss(outputs, targets)
            running_psnr_loss += psnr_loss_value.item()

    running_loss = torch.tensor(running_loss, device=device)
    avg_loss = torch.sqrt(running_loss / len(val_loader))
    avg_nmse_loss = running_nmse_loss / len(val_loader)
    avg_ssim_loss = running_ssim_loss / len(val_loader)
    avg_psnr_loss = running_psnr_loss / len(val_loader)

    print(f"val Loss: {avg_loss:.4f}")
    print(f"val avg_nmse_loss: {avg_nmse_loss:.4f}")
    print(f"val avg_ssim_loss: {avg_ssim_loss:.4f}")
    print(f"val avg_psnr_loss: {avg_psnr_loss:.4f}")
    # Write validation metrics to TensorBoard
    writer.add_scalar('Loss/val', avg_loss, epoch)



def train(model, train_loader, val_loader, num_epochs, device, save_interval=5):

    log_dir = r'../runs/model_log/BTM_ghost_net'
    # 清空 log_dir 下的文件（如果存在）
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除整个目录及其内容
    # 重新创建 log_dir
    os.makedirs(log_dir)


    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard SummaryWriter
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
            torch.save(model.state_dict(), f"../runs/model_pth/BTM_ghost_net/checkpoint_epoch_{epoch+1}.pth")

    writer.close()


if __name__ == '__main__':
    # import psutil
    # process = psutil.Process()
    # #设置CPU限制
    # process.nice(psutil.IDLE_PRIORITY_CLASS)
    # torch.set_num_threads(1)

    Radio_train = RadioUNet_c_sprseIRT4(phase="train", carsSimul="yes", carsInput="yes")
    Radio_val = RadioUNet_c_sprseIRT4(phase="val", carsSimul="yes", carsInput="yes")
    Radio_test = RadioUNet_c_sprseIRT4(phase="test", carsSimul="yes", carsInput="yes")
    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    train_batch_size = 8  # 批次大小
    test_batch_size = 8  # 批次大小

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=train_batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(Radio_val, batch_size=test_batch_size, shuffle=True, num_workers=4)
    }

    # 设置设备为GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cpu')

    BTM_ghost_UNet_input_shape = [4, 256, 256]
    BTM_ghost_UNet_output_shape = [1, 256, 256]
    C_down_list = [32, 64, 128, 256]
    C_list_attn = torch.tensor([64, 64, 64, 128, 128, 128, 128])
    net = BTM_ghost_UNet(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,C_list_attn).to(device)
    net.load_weights("../runs/model_pth/BTM_ghost_net/checkpoint_epoch_130.pth")
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # 开始训练
    train(net, train_loader, val_loader, num_epochs=400, device=device, save_interval=5)
