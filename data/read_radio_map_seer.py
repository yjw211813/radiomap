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
from lib import loaders, modules


if __name__ == '__main__':
    Radio_train = loaders.RadioUNet_c_sprseIRT4(phase="train")
    Radio_val = loaders.RadioUNet_c_sprseIRT4(phase="val")
    Radio_test = loaders.RadioUNet_c_sprseIRT4(phase="test")

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

    # 显示结果
    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    plt.imshow(image_gain[0])
    plt.title('Modified Gain[0]')

    plt.subplot(232)
    plt.imshow(image_sample[0])
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
    plt.imshow(mask, cmap='gray')  # 显示掩码区域
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