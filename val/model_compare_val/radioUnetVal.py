import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict  # 用于创建带默认值的字典
import torch.nn.functional as F
import torch.nn as nn
from model.radioUnetModel import modules  # 导入自定义模型模块
from torch.utils.data import Dataset, DataLoader
import os
from data.lib import loaders  # 导入数据加载器

# 模型保存路径
model_save_dir = "/home/code/radio_map_construction/runs/model_compare/radioUnet/"


def calc_loss_test(pred1, pred2, target, metrics, error="MSE"):
    """计算测试损失并更新指标字典

    参数:
        pred1 (Tensor): 第一个预测输出
        pred2 (Tensor): 第二个预测输出
        target (Tensor): 真实标签
        metrics (dict): 存储指标的字典
        error (str): 损失类型，支持 'MSE' 或 'NMSE'

    返回:
        list: 包含两个损失的列表 [loss1, loss2]
    """
    criterion = nn.MSELoss()  # 均方误差损失函数

    # 根据误差类型计算损失
    if error == "MSE":
        loss1 = criterion(pred1, target)
        loss2 = criterion(pred2, target)
    else:  # NMSE (归一化MSE)
        loss1 = criterion(pred1, target) / criterion(target, torch.zeros_like(target))
        loss2 = criterion(pred2, target) / criterion(target, torch.zeros_like(target))

    # 更新指标字典（累加批次损失）
    metrics['loss first U'] += loss1.item() * target.size(0)  # item()获取标量值
    metrics['loss second U'] += loss2.item() * target.size(0)

    return [loss1, loss2]


def print_metrics_test(metrics, epoch_samples, error):
    """打印测试指标

    参数:
        metrics (dict): 累计的指标字典
        epoch_samples (int): 总样本数
        error (str): 当前计算的误差类型
    """
    outputs = []
    # 计算每个指标的平均值
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    # 打印格式化结果
    print("{}: {}".format("Test" + " " + error, ", ".join(outputs)))


def test_loss(device, model, Radio_test, batch_size, error="MSE", dataset="coarse"):
    """测试模型性能

    参数:
        device (torch.device): 计算设备 (CPU/GPU)
        model (nn.Module): 待测试的模型
        Radio_test (Dataset): 测试数据集
        batch_size (int): 批次大小
        error (str): 损失类型 ('MSE'/'NMSE')
        dataset (str): 数据集类型 ('coarse'/'fine')
    """
    since = time.time()  # 开始计时
    model.eval()  # 设置模型为评估模式（关闭dropout等）
    metrics = defaultdict(float)  # 初始化指标字典
    epoch_samples = 0  # 总样本计数器

    # 根据数据集类型选择数据加载方式
    if dataset == "coarse":
        dataloader = DataLoader(Radio_test, batch_size=batch_size, shuffle=True,
                                num_workers=1, generator=torch.Generator(device=device))
    elif dataset == "fine":
        dataloader = DataLoader(Radio_test, batch_size=batch_size, shuffle=True, num_workers=1, generator=torch.Generator(device=device))

    # 禁用梯度计算以加速测试
    with torch.no_grad():
        for batch in dataloader:
            # 解包批次数据（不同数据集结构不同）
            if dataset == "coarse":
                inputs, targets = batch
            else:  # fine数据集有三个返回值
                inputs, targets, samples = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs1, outputs2 = model(inputs)

            # 计算损失并更新指标
            loss1, loss2 = calc_loss_test(outputs1, outputs2, targets, metrics, error)
            epoch_samples += inputs.size(0)  # 累加样本数

    # 打印指标
    print_metrics_test(metrics, epoch_samples, error)

    # 计算并打印总耗时
    time_elapsed = time.time() - since
    print('测试耗时: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# 主程序入口
if __name__ == "__main__":
    # 1. 加载测试数据集
    Radio_test = loaders.RadioUNet_c_sprseIRT4(phase="test")

    # 2. 设置超参数
    batch_size = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. 初始化模型（选择第二阶段U-Net）
    model = modules.RadioWNet(phase="secondU")
    model.load_state_dict(os.path.join(model_save_dir, "Trained_Model_SecondU.pt"))
    model.to(device)  # 将模型移至计算设备


    # 5. 执行测试（MSE和NMSE两种指标）
    test_loss(device, model, Radio_test, batch_size, error="MSE")
    test_loss(device, model, Radio_test, batch_size, error="NMSE")