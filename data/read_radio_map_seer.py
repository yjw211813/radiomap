from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#from lib import RadioUNet_modules3, RadioUNet_loaders2
from data.lib.loaders import RadioMapSeerLoader


if __name__ == '__main__':
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

    My_Radio_train = RadioMapSeerLoader(simuSetDict, phase="train")
    i = 400
    inputs, image_gain = My_Radio_train[i]
    mask = inputs[4] != 0
    image_gain[0][mask] += image_gain[0][mask]

    # 显示结果
    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    plt.imshow(image_gain[0], cmap='jet')
    plt.title('Modified Gain[0]')

    plt.subplot(232)
    plt.imshow(inputs[0])
    plt.title('Sample[0]')

    plt.subplot(233)
    plt.imshow(inputs[1])
    plt.title('Build_ant[0]')

    plt.subplot(234)
    plt.imshow(inputs[2], cmap='jet')
    plt.title('Build_ant[1]')

    plt.subplot(235)
    plt.imshow(inputs[3], cmap='jet')
    plt.title('Build_ant[2] (Mask Source)')

    plt.subplot(236)
    plt.imshow(inputs[4])  # 显示掩码区域
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