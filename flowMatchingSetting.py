import torch
import shutil
import os
from model_app.flowMatchingApp import flowMatching_app
from torch.utils.tensorboard import SummaryWriter
from data.lib.loaders import RadioMapSeerLoader
from torch.utils.data import DataLoader
from model.flow_matching_model.volecity_predict import velocity_UNet
from data.radioSeerRead import create_dataloaders

if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 16  # 批次大小
    val_batch_size = 2
    test_batch_size = 2   # 批次大小

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
        "fix_samples": 300,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
        "num_samples_low": 10,  # 最低采样数
        "num_samples_high": 300,  # 最高采样数
        "inter_flag":True, # 看是否需要插值图像
        "scale256_flag": False# 取值范围是否为0 - 255
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
    # 训练标识
    print("flowMatching_MSAA")
    #   模型定义参数类
    # if simuSetDict["carsInput"] !="no":
    #     UNet_BTM_input_shape = [6, 256, 256]
    # else:
    #     UNet_BTM_input_shape = [5, 256, 256]
    net_info_dict = {
        "T": 100,
        "in_shape": [train_batch_size, 6, 256, 256],
        "out_shape": [train_batch_size, 1, 256, 256],
        "C_list": [64, 128, 256, 512],
        "attn_list": ["LSKNet", "LSKNet", "Attn", "Attn"],
        "conv_kernels": [3, 5, 7, 9],
        "conv_dilats": [1, 1, 1, 1],
        "LSK_kernels": [5, 7, 5, 5],
        "LSK_dilats": [1, 3, 1, 1],
        "LSK_mid_kernel": 7,
        "encoderDownKernels": [3, 5, 7,9],
        "fra_kernels": [3, 5, 7,9],
        "fra_dilates": [1, 1, 1,1],
        "MSAA_pool_kernel": 7,
        "MSAA_kernels": [3, 5, 7,9],
        "MSAA_dilats": [1, 1, 1,1],
        "decoderUpKernels": [3, 5, 7,9],
        "tdim": int(512*4),
        "tail_kernel": 5
    }
    model = velocity_UNet(net_info_dict)

    total_epoch = 100
    start_epoch = 4
    #   定义训练过程数据保存地址
    log_dir = r'/home/code/radio_map_construction/runs/model_log/flow_matching_MSAA/'# log 存储位置
    model_load_dir = r"/home/code/radio_map_construction/runs/model_pth/flow_matching_MSAA/"# 模型加载目录
    model_save_dir = r"/home/code/radio_map_construction/runs/model_pth/flow_matching_MSAA/"# 模型存储位置
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除上一次训练过程数据
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    val_dir = r"/home/code/radio_map_construction/runs/model_val_log/flow_matching_MSAA/"
    print(val_dir)
    app = flowMatching_app(start_epoch = start_epoch,
                           model_save_dir = model_save_dir,
                           model_load_dir = model_load_dir,
                           writer = writer,
                           device = device,
                           T = net_info_dict["T"])
    # app.train(model, train_loader, val_loader,val_dir, total_epoch)
    load_epoch = 28
    test_dir = r"/home/code/radio_map_construction/runs/model_test_log/flow_matching_MSAA/"
    os.makedirs(test_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_load_dir, f"checkpoint_epoch_{load_epoch}.pth")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("model load weight done.")
    app.val(model, val_loader, val_dir)



