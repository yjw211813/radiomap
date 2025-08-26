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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 32  # 批次大小
    val_batch_size = 32
    test_batch_size = 8  # 批次大小

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
    print("flowMatching")
    #   模型定义参数类
    if simuSetDict["carsInput"] !="no":
        UNet_BTM_input_shape = [6, 256, 256]
    else:
        UNet_BTM_input_shape = [5, 256, 256]

    UNet_BTM_output_shape = [1, 256, 256]
    C_down_list =  [32, 64, 128, 256]
    C_list_attn = torch.tensor([64, 64, 64, 128, 128, 128, 128])
    attn_params = [C_list_attn * 2, C_list_attn , C_list_attn // 2, C_list_attn // 2]
    T = 100
    #   定义训练过程数据保存地址
    log_dir = r'/home/code/radio_map_construction/runs/model_log/flow_matching01/'# log 存储位置
    model_load_dir = r"/home/code/radio_map_construction/runs/model_pth/flow_matching01/"# 模型加载目录
    model_save_dir = r"/home/code/radio_map_construction/runs/model_pth/flow_matching01/"# 模型存储位置
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除上一次训练过程数据
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)


    model = velocity_UNet(T = T,                                           # 流匹配的迭代总轮次
                           input_shape = UNet_BTM_input_shape,                   # 网络输入形状
                           output_shape = UNet_BTM_output_shape,                 # 网络输出形状
                           C_down_list = C_down_list,                            # Unet网络通道设置列表
                           attn_params = attn_params).to(device)                 # 掩码网络通道设置列表


    total_epoch = 300
    start_epoch = 10
    val_dir = r"/home/code/radio_map_construction/runs/model_val_log/flow_matching01/"
    print(val_dir)
    app = flowMatching_app(start_epoch = start_epoch,
                           model_save_dir = model_save_dir,
                           model_load_dir = model_load_dir,
                           writer = writer,
                           device = device,
                           T = T)
    load_epoch = 33
    test_dir = r"/home/code/radio_map_construction/runs/model_test_log/flow_matching01/"
    os.makedirs(test_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_load_dir, f"checkpoint_epoch_{load_epoch}.pth")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("model load weight done.")
    app.val(model, val_loader, val_dir)



