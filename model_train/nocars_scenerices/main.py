import torch
from pandas.tests.tseries.frequencies.test_inference import base_delta_code_pair

from model_app.SAUnet_app import SAUnet_app
from model_train.data_config import get_cars_load, get_nocars_load
from model.sigle_Unet.SAUnet import SAUnet
import os

if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)

    train_loader, val_loader, test_loader = get_nocars_load()
    base_path = r"/home/code/radioMap"
    log_dir = base_path + r"/runs/model_log/old_SAUnet/"# log 存储位置
    model_load_dir = base_path + r"/runs/model_pth/old_SAUnet/"# 模型加载目录
    model_save_dir = base_path + r"/runs/model_pth/old_SAUnet/"# 模型存储位置

    os.makedirs(model_save_dir, exist_ok=True)

    print("old_SAUnet,cuda = 3")
    # 定义模型

    BTM_ghost_UNet_input_shape = [6, 256, 256]
    BTM_ghost_UNet_output_shape = [1, 256, 256]
    C_down_list = [32, 64, 128, 256]
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    model = SAUnet(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params)

    # 定义训练对象
    warmup_epochs = 5
    total_epoch = 100
    start_epoch = 0

    app = SAUnet_app(start_epoch,log_dir,warmup_epochs,model_save_dir,device)
    load_epoch = 0
    val_dir = base_path + r"/runs/model_val_log/old_SAUnet/"
    app.train(model, train_loader, val_loader, total_epoch)
