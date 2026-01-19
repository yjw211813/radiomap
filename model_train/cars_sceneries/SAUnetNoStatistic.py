import torch
from model_app.SAUnet_app import SAUnet_app
from model.sigle_Unet.SAUnet import SAUnet
import os
from model_train.data_config import get_cars_load

if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)

    train_loader, val_loader, test_loader = get_cars_load()
    base_dir = r"/home/code/radioMap/runs/"
    log_dir = base_dir + r'model_log/old_SAUnetNoSatistic/'# log 存储位置
    model_load_dir = base_dir + r"model_pth/old_SAUnetNoSatistic/"# 模型加载目录
    model_save_dir = base_dir + r"model_pth/old_SAUnetNoSatistic/"# 模型存储位置

    os.makedirs(model_save_dir, exist_ok=True)

    print("old_SAUnetNoSatistic,cuda = 2")
    # 定义模型

    BTM_ghost_UNet_input_shape = [5, 256, 256]
    BTM_ghost_UNet_output_shape = [1, 256, 256]
    C_down_list = [64, 128, 256, 512]
    C_list_attn = torch.tensor([64, 64, 128, 128, 256])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    model = SAUnet(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params)

    # 定义训练对象
    warmup_epochs = 5
    total_epoch = 100
    start_epoch = 0

    app = SAUnet_app(start_epoch,log_dir,warmup_epochs,model_save_dir,device)
    load_epoch = 0
    val_dir = base_dir + r"model_val_log/old_SAUnetNoSatistic/"
    app.train(model, train_loader, val_loader, total_epoch)
