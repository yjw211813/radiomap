from __future__ import print_function, division
import os
import torch
import time
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial import cKDTree
from data.lib.loaders import RadioMapSeerLoader


def idw_interpolation_tensor(matrix, k=5, power=2):
    """
    Tensor版本的反距离加权插值
    :param matrix: 输入张量，形状(1, H, W)
    :param k: 使用的最近邻点数量
    :param power: 距离权重指数
    :return: 插值后的张量
    """
    # 确保输入是CPU上的numpy数组进行处理
    if matrix.is_cuda:
        matrix_cpu = matrix.detach().cpu().clone()
    else:
        matrix_cpu = matrix.detach().clone()

    # 转换为numpy数组处理
    data_np = matrix_cpu.numpy()[0]  # 去掉批次维度
    height, width = data_np.shape

    # 获取所有非零点的坐标和值
    non_zero_mask = data_np != 0
    non_zero_coords = np.argwhere(non_zero_mask)
    non_zero_values = data_np[non_zero_mask]

    if len(non_zero_coords) == 0:
        return matrix_cpu

    # 构建KDTree加速最近邻搜索
    tree = cKDTree(non_zero_coords)

    # 获取所有零值点坐标
    zero_coords = np.argwhere(data_np == 0)

    # 批量查询所有零值点的k个最近邻
    distances, indices = tree.query(zero_coords, k=k)

    # 避免除以零错误
    distances = np.maximum(distances, 1e-12)

    # 计算权重 (1/d^power)
    weights = 1 / (distances ** power)

    # 获取对应的非零值
    neighbor_values = non_zero_values[indices]

    # 计算加权平均值
    weighted_sum = np.sum(weights * neighbor_values, axis=1)
    sum_weights = np.sum(weights, axis=1)
    interpolated_values = weighted_sum / sum_weights

    # 更新零值点
    for (y, x), value in zip(zero_coords, interpolated_values):
        data_np[y, x] = value

    # 转换回PyTorch Tensor
    result = torch.from_numpy(data_np).unsqueeze(0)  # 重新添加批次维度

    # 如果原始输入在GPU上，将结果移回GPU
    if matrix.is_cuda:
        result = result.to(matrix.device)

    return result


import numpy as np

import torch


def batch_sample_images_torch(input_tensor, samples_per_image=4, fix_samples=0,
                              num_samples_low=100, num_samples_high=200):
    """
    PyTorch优化的批量图像采样函数

    参数:
    input_tensor: 输入张量，形状为(B, C, H, W)
    samples_per_image: 每张图像的采样次数 (默认4)
    fix_samples: 固定采样点数 (0表示随机)
    num_samples_low: 随机采样点数下限 (默认100)
    num_samples_high: 随机采样点数上限 (默认200)

    返回:
    output: 采样后的张量，形状为(B*samples_per_image, C, H, W)
    """
    device = input_tensor.device
    B, C, H, W = input_tensor.shape
    total_samples = B * samples_per_image

    # 预计算所有样本的采样点数
    if fix_samples == 0:
        num_samples_arr = torch.randint(num_samples_low, num_samples_high,
                                        (total_samples,), device=device)
    else:
        num_samples_arr = torch.full((total_samples,), fix_samples,
                                     device=device, dtype=torch.int32)

    # 创建输出张量
    output = torch.zeros((total_samples, C, H, W), device=device)

    # 处理每个输出样本
    for idx in range(total_samples):
        # 确定对应的原始图像
        b = idx // samples_per_image
        img = input_tensor[b, 0]  # (H, W)

        # 获取当前样本的采样点数
        num_samples = num_samples_arr[idx].item()

        # 生成随机采样点坐标
        x_samples = torch.randint(0, H, (num_samples,), device=device)
        y_samples = torch.randint(0, W, (num_samples,), device=device)

        # 使用高级索引直接赋值
        output[idx, 0, x_samples, y_samples] = img[x_samples, y_samples]

    return output

if __name__ == '__main__':
    # 创建模拟输入 (9, 1, 256, 256)
    input_data = torch.rand(9, 1, 256, 256).cuda()

    # 获取采样结果 (36, 1, 256, 256)
    sampled_data = batch_sample_images_torch(
        input_data,
        samples_per_image=4,  # 每张图采样4次
        fix_samples=150  # 固定150个采样点
    )

    print("输入形状:", input_data.shape)  # (9, 1, 256, 256)
    print("输出形状:", sampled_data.shape)  # (36, 1, 256, 256)
    print(sampled_data.device)
    # simuSetDict = {
    #     "ind1": 0,                                                           # 起始索引
    #     "ind2": 0,                                                           # 末尾索引
    #     "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/", # 数据集文件夹
    #     "numTx": 80,                                                         # 信源数量设定
    #     "thresh": 0.05,                                                      # 环境噪声
    #     "simulation": "IRT2",                      # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
    #     "carsSimul": "yes",                        # 是否开启小车作为仿真
    #     "carsInput": "yes",                        # 是否将小车图作为模型输入
    #     "IRT2maxW": 1,                            # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
    #     "cityMap": "complete",                    # 是否输入完全的城市地图
    #     "missing": 1,                             # 地图缺失号码
    #     "fix_samples": 300,                         # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
    #     "num_samples_low": 10,                    # 最低采样数
    #     "num_samples_high": 300                   # 最高采样数
    # }
    #
    # My_Radio_train = RadioMapSeerLoader(simuSetDict,phase="train")
    #
    #
    # i=800
    # inputs, image_gain = My_Radio_train[i]
    # mask = inputs[3] != 0
    # image_gain[0][mask] += image_gain[0][mask]
    #
    # # 在掩码位置叠加image_gain[0]的值 (增强效果)
    # # 注意：这里直接加到原始image_gain[0]上，会修改原始数据
    # # 如需保留原始数据，应先复制: modified_gain = image_gain[0].copy()
    # image_gain[0][mask] += image_gain[0][mask]  # 翻倍增强
    # image_sample = inputs[2].unsqueeze(0)
    # print("原始矩阵零值点数量:", torch.sum(image_sample == 0.0).item())
    # # 执行IDW插值
    #
    # start_time = time.time()
    # interpolated_matrix = idw_interpolation_tensor(image_sample, k=5, power=2)
    #
    # end_time = time.time()
    #
    # execution_time = end_time - start_time
    # print(f"IDW插值执行时间: {execution_time:.4f} 秒")
    # print("插值后零值点数量:", torch.sum(interpolated_matrix == 0.0).item())
    #
    # # 显示结果
    # plt.figure(figsize=(15, 10))
    #
    # plt.subplot(231)
    # plt.imshow(image_gain[0], cmap='jet')
    # plt.title('Modified Gain[0]')
    #
    # plt.subplot(232)
    # plt.imshow(inputs[0])
    # plt.title('Sample[0]')
    #
    # plt.subplot(233)
    # plt.imshow(inputs[1])
    # plt.title('Build_ant[0]')
    #
    # plt.subplot(234)
    # plt.imshow(inputs[2], cmap='jet')
    # plt.title('Build_ant[1]')
    #
    # plt.subplot(235)
    # plt.imshow(inputs[3])
    # plt.title('Build_ant[2] (Mask Source)')
    #
    # plt.subplot(236)
    # plt.imshow(interpolated_matrix[0],cmap='jet')  # 显示掩码区域
    # plt.title('Mask Region')
    #
    # plt.tight_layout()
    # plt.show()
'''
    nonzero_mask = (image_sample != 0)

    # 提取这些位置的 image_sample 和 image_gain 的值
    sample_values = image_sample[nonzero_mask]
    gain_values = image_gain[nonzero_mask]

    # 检查是否所有对应的值都相等
    are_equal = np.allclose(sample_values, gain_values)

    if are_equal:
        print("所有 image_sample 中不为 0 的位置的值都等于 image_gain 中对应位置的值。")
    else:
        print("存在 image_sample 中不为 0 的位置的值不等于 image_gain 中对应位置的值。")
'''