import torch
from model_app.Unet_BTM_app import Unet_BTM_app
from data.lib.loaders import RadioMapSeerLoader
from torch.utils.data import DataLoader
from model.sigle_Unet.Unet_BTM import Unet_BTM
import os



if __name__ == '__main__':



    simuSetDict = {
        "ind1": 0,  # 起始索引
        "ind2": 0,  # 末尾索引
        "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",  # 数据集文件夹
        "numTx": 80,  # 信源数量设定
        "thresh": 0.05,  # 环境噪声
        "simulation": "rand",  # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
        "carsSimul": "no",  # 是否开启小车作为仿真
        "carsInput": "no",  # 是否将小车图作为模型输入
        "IRT2maxW": 0.3,  # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
        "cityMap": "complete",  # 是否输入完全的城市地图
        "missing": 1,  # 地图缺失号码
        "fix_samples": 300,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
        "num_samples_low": 10,  # 最低采样数
        "num_samples_high": 300,  # 最高采样数
        "inter_flag":True # 看是否需要插值图像
    }

    # 训练标识
    print("inter_flag:",simuSetDict["inter_flag"])
    print("MS_no_cars")


    train_batch_size = 32  # 批次大小
    val_batch_size = 32
    test_batch_size = 8  # 批次大小
    warmup_epochs = 4
    total_epoch = 80
    if simuSetDict["carsInput"] !="no":
        BTM_ghost_UNet_input_shape = [5, 256, 256]
    else:
        BTM_ghost_UNet_input_shape = [4, 256, 256]
    BTM_ghost_UNet_output_shape = [1, 256, 256]
    C_down_list =  [32, 64, 128, 256]
    C_list_attn = torch.tensor([64, 64, 64, 128, 128, 128, 128])
    attn_params = [C_list_attn * 2, C_list_attn , C_list_attn // 2, C_list_attn // 2]
    log_dir = r'/home/code/radio_map_construction/runs/model_log/UNet_MS_no_cars/'# log 存储位置
    model_load_dir = r"/home/code/radio_map_construction/runs/model_pth/UNet_MS_no_cars/"# 模型加载目录
    model_save_dir = r"/home/code/radio_map_construction/runs/model_pth/UNet_MS_no_cars/"# 模型存储位置
    start_epoch = 0
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # 定义训练对象
    app = Unet_BTM_app(start_epoch,log_dir,warmup_epochs,model_save_dir)


    # 加载数据集
    Radio_train = RadioMapSeerLoader(simuSetDict, phase="train")
    Radio_val = RadioMapSeerLoader(simuSetDict, phase="val")
    Radio_test = RadioMapSeerLoader(simuSetDict, phase="test")

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=train_batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(Radio_val, batch_size=val_batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(Radio_test, batch_size=test_batch_size, shuffle=True, num_workers=4)
    }

    # 定义模型
    model = Unet_BTM(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,attn_params).to(device)

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    os.makedirs(model_save_dir, exist_ok=True)

    load_epoch = 38
    val_dir = r"/home/code/radio_map_construction/runs/model_val_log/UNet_MS_no_cars/"
    app.test( model, load_epoch, test_loader, device, val_dir)
