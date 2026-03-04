import torch
from model_app.UVM_app import UVM_app
from model.UVM.UVM_model import UVMNet
import os

from model_train.data_config import get_cars_load

if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    base_dir = r"/home/code/radioMap/runs/"

    train_loader, val_loader, test_loader = get_cars_load()

    log_dir = base_dir + r'model_log/UVM_cars/'  # log 存储位置
    model_load_dir = base_dir + r"model_pth/UVM_cars/"  # 模型加载目录
    model_save_dir = base_dir + r"model_pth/UVM_cars/"  # 模型存储位置


    os.makedirs(model_save_dir, exist_ok=True)
    print("UVM_cars train")
    model = UVMNet(n_channels = 6)
    warmup_epochs = 10
    total_epoch = 80
    start_epoch = 20

    app = UVM_app(start_epoch, log_dir, warmup_epochs, model_save_dir, device)
    load_epoch = 20
    val_dir = base_dir + r"model_val_log/UVM_cars/"
    app.train(model, train_loader, val_loader, total_epoch)

    # app.test(model, load_epoch, test_loader, val_dir)



