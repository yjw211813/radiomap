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
from data.lib.loaders import RadioUNet_c_sprseIRT4


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
        matrix_cpu = matrix.cpu()
    else:
        matrix_cpu = matrix

    # 转换为numpy数组处理
    data_np = matrix_cpu.numpy()[0]  # 去掉批次维度
    height, width = data_np.shape

    # 获取所有非零点的坐标和值
    non_zero_mask = data_np != 0
    non_zero_coords = np.argwhere(non_zero_mask)
    non_zero_values = data_np[non_zero_mask]

    # 如果没有非零点，直接返回原矩阵
    if len(non_zero_coords) == 0:
        return matrix

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



if __name__ == '__main__':
    Radio_train = RadioUNet_c_sprseIRT4(phase="train")
    Radio_val = RadioUNet_c_sprseIRT4(phase="val")
    Radio_test = RadioUNet_c_sprseIRT4(phase="test")

    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    batch_size = 15

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1),
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1)
    }

    i=800
    image_build_ant, image_gain,image_sample = Radio_train[i]
    image_sample = image_sample * image_gain

    # 创建掩码：标记image_build_ant[2]中非零像素的位置
    mask = image_build_ant[2] != 0

    # 在掩码位置叠加image_gain[0]的值 (增强效果)
    # 注意：这里直接加到原始image_gain[0]上，会修改原始数据
    # 如需保留原始数据，应先复制: modified_gain = image_gain[0].copy()
    image_gain[0][mask] += image_gain[0][mask]  # 翻倍增强
    print("原始矩阵零值点数量:", torch.sum(image_sample == 0.0).item())
    # 执行IDW插值

    start_time = time.time()
    interpolated_matrix = idw_interpolation_tensor(image_sample, k=5, power=2)

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"IDW插值执行时间: {execution_time:.4f} 秒")
    print("插值后零值点数量:", torch.sum(interpolated_matrix == 0.0).item())

    # 显示结果
    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    plt.imshow(image_gain[0], cmap='jet')
    plt.title('Modified Gain[0]')

    plt.subplot(232)
    plt.imshow(interpolated_matrix[0], cmap='jet')
    plt.title('Sample[0]')

    plt.subplot(233)
    plt.imshow(image_build_ant[0])
    plt.title('Build_ant[0]')

    plt.subplot(234)
    plt.imshow(image_build_ant[1])
    plt.title('Build_ant[1]')

    plt.subplot(235)
    plt.imshow(image_build_ant[2])
    plt.title('Build_ant[2] (Mask Source)')

    plt.subplot(236)
    plt.imshow(image_sample[0], cmap='jet')  # 显示掩码区域
    plt.title('Mask Region')

    plt.tight_layout()
    plt.show()

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