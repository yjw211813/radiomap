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

import os


log_dir = r'/home/code/radio_map_construction/runs/model_log/model_compare/radioUnet/'
# model_load_dir = "/home/code/radio_map_construction/runs/model_pth/BTM_multi_scale_v6_ssim/"
model_save_dir = "/home/code/radio_map_construction/runs/model_compare/radioUnet/"

def calc_loss_dense(pred, target, metrics):
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def calc_loss_sparse(pred, target, samples, metrics, num_samples):
    criterion = nn.MSELoss()
    loss = criterion(samples*pred, samples*target)*(256**2)/num_samples
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    outputs2 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs1)))

def train_model(device,model, optimizer, scheduler,dataloaders, num_epochs=50, WNetPhase="firstU", targetType="dense", num_samples=300):
    # WNetPhase: traine first U and freez second ("firstU"), or vice verse ("secondU").
    # targetType: train against dense images ("dense") or sparse measurements ("sparse")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("learning rate", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            if targetType=="dense":
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        [outputs1,outputs2] = model(inputs)
                        if WNetPhase=="firstU":
                            loss = calc_loss_dense(outputs1, targets, metrics)
                        else:
                            loss = calc_loss_dense(outputs2, targets, metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)
            elif targetType=="sparse":
                for inputs, targets, samples in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    samples = samples.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        [outputs1,outputs2] = model(inputs)
                        if WNetPhase=="firstU":
                            loss = calc_loss_sparse(outputs1, targets, samples, metrics, num_samples)
                        else:
                            loss = calc_loss_sparse(outputs2, targets, samples, metrics, num_samples)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 读取数据
    Radio_train = loaders.RadioUNet_c(phase="train")
    Radio_val = loaders.RadioUNet_c(phase="val")
    Radio_test = loaders.RadioUNet_c(phase="test")

    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    batch_size = 15

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1,generator=torch.Generator(device=device) ),
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1,generator=torch.Generator(device=device) )
    }

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.enabled
    model =modules.RadioWNet(phase="firstU")
    model.to(device)
    summary(model, input_size=(2, 256,256))

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(device,model, optimizer_ft, exp_lr_scheduler,dataloaders)

    try:
        os.mkdir(model_save_dir)
    except OSError as error:
        print(error)
    # 保存第一个UNet的权重
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"Trained_Model_FirstU.pt"))
 
    model =modules.RadioWNet(phase="secondU")
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f"Trained_Model_FirstU.pt")))
    model.to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    model = train_model(device,model, optimizer_ft, exp_lr_scheduler,dataloaders, WNetPhase="secondU")
    
    torch.save(model.state_dict(),os.path.join(model_save_dir, f"Trained_Model_SecondU.pt"))








