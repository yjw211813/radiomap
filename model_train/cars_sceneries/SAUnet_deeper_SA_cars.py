import torch

from model_app.SAUnet_app import SAUnet_app
# from model.sigle_Unet.simple_CNN import SAUnetForProcess
from model.sigle_Unet.SAUnet_deeper_BTM import SAUnet_deeper_BTM
import os
from model_train.data_config import get_cars_load


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)

    train_loader, val_loader, test_loader =  get_cars_load()
    base_dir = r"/home/code/radioMap/runs/"
    log_dir = base_dir + r'model_log/SAUnet_deeper_BTM_cars/'# log 存储位置
    model_load_dir = base_dir + r"model_pth/SAUnet_deeper_BTM_cars/"# 模型加载目录
    model_save_dir = base_dir + r"model_pth/SAUnet_deeper_BTM_cars/"# 模型存储位置

    os.makedirs(model_save_dir, exist_ok=True)
    print("SAUnet,cuda = 0")
    # 定义模型

    input_shape = [6, 256, 256]
    output_shape = [1, 256, 256]
    C_down_list = [32, 64, 128, 256]
    C_list_attn = torch.tensor([64, 64, 128, 128, 256])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]
    model = SAUnet_deeper_BTM(input_shape = input_shape,output_shape= output_shape,
                   C_down_list=C_down_list, attn_params=attn_params)


    # 定义训练对象
    warmup_epochs = 5
    total_epoch = 100
    start_epoch = 0

    app = SAUnet_app(start_epoch,log_dir,warmup_epochs,model_save_dir,device)
    load_epoch = 0
    val_dir = r"/home/code/radioMap/runs/model_val_log/SAUnet_deeper_BTM_cars/"
    app.train(model, train_loader, val_loader, total_epoch)
