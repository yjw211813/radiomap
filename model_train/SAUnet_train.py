import torch
from model_app.Unet_BTM_app import Unet_BTM_app
from data.lib.seer_loader import RadioMapSeerLoader
from torch.utils.data import DataLoader
from model.sigle_Unet.simple_CNN import SAUnetForProcess
import os

if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    train_batch_size = 16  # 批次大小
    val_batch_size = 16
    test_batch_size = 16  # 批次大小1
    
    simuSetDict = {
        "ind1": 0,  # 起始索引
        "ind2": 0,  # 末尾索引
        "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",  # 数据集文件夹
        "numTx": 80,  # 信源数量设定
        "thresh": 0.05,  # 环境噪声
        "simulation": "rand",  # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
        "carsSimul": "yes",  # 是否开启小车作为仿真
        "carsInput": "yes",  # 是否将小车图作为模型输入
        "IRT2maxW": 0.3,  # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
        "cityMap": "complete",  # 是否输入完全的城市地图
        "missing": 1,  # 地图缺失号码
        "fix_samples": 655,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
        "num_samples_low": 10,  # 最低采样数
        "num_samples_high": 300,  # 最高采样数
        "inter_flag": True,  # 看是否需要插值图像
        "scale256_flag": True,  # 取值范围是否为0 - 255
        "sample_flag": True,  # 是否有采样输入
        "loss_samples_flag": False,  # 是否定义loss为稀疏采样loss
        "formula_flag": True
    }
    # 加载数据集
    Radio_train = RadioMapSeerLoader(simuSetDict, phase="train")
    Radio_val = RadioMapSeerLoader(simuSetDict, phase="val")
    Radio_test = RadioMapSeerLoader(simuSetDict, phase="test")

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=train_batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(Radio_val, batch_size=val_batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(Radio_test, batch_size=test_batch_size, shuffle=True, num_workers=4)
    }
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    log_dir = r'/home/code/radio_map_construction/runs/model_log/SAUnetForProcess/'# log 存储位置
    model_load_dir = r"/home/code/radio_map_construction/runs/model_pth/SAUnetForProcess/"# 模型加载目录
    model_save_dir = r"/home/code/radio_map_construction/runs/model_pth/SAUnetForProcess/"# 模型存储位置

    os.makedirs(model_save_dir, exist_ok=True)

    print("SAUnetForProcess,cuda = 3")
    # 定义模型

    input_shape = [6, 256, 256]
    output_shape = [1, 256, 256]
    model = SAUnetForProcess(input_shape = input_shape,output_shape= output_shape)

    # 定义训练对象
    warmup_epochs = 5
    total_epoch = 100
    start_epoch = 0

    app = Unet_BTM_app(start_epoch,log_dir,warmup_epochs,model_save_dir,device)
    load_epoch = 0
    val_dir = r"/home/code/radio_map_construction/runs/model_val_log/SAUnetForProcess/"
    app.train(model, train_loader, val_loader, total_epoch)
