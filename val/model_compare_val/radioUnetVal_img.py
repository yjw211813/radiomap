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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform

# 模型保存路径
model_save_dir = "/home/code/radio_map_construction/runs/model_compare/radioUnet/"
img_save_dir = "/home/code/radio_map_construction/runs/model_compare/radioUnet/example/"


# 主程序入口
if __name__ == "__main__":
    # 1. 加载测试数据集
    # Radio_test = loaders.RadioUNet_c_sprseIRT4(phase="test")

    try:
        os.mkdir(img_save_dir)
    except OSError as error:
        print(error)
    # 2. 设置超参数
    batch_size = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. 初始化模型（选择第二阶段U-Net）
    model = modules.RadioWNet(phase="secondU")
    model.load_state_dict(os.path.join(model_save_dir, "Trained_Model_SecondU.pt"))
    model.to(device)  # 将模型移至计算设备

    maps_inds=np.arange(0,700,1,dtype=np.int16)
    #Standard determenistic "random" shuffle of the maps:
    np.random.seed(42)
    np.random.shuffle(maps_inds)

    for mapp in range(1, 99):
        name00 = str(mapp)
        name0 = str(maps_inds[600 + mapp])
        Radio_test2 = loaders.RadioUNet_c(maps_inds=maps_inds, phase="custom",
                                          ind1=600 + mapp, ind2=600 + mapp)
        ii = 0
        for inputs, target in DataLoader(Radio_test2, batch_size=1, shuffle=False, num_workers=0):
            ii = ii + 1
            if ii > 2:
                break
            inputs = inputs.to(device)
            target = target.to(device)
            pred1, pred = model(inputs)
            pred = (256 * pred.detach().cpu().numpy()).astype(np.uint8)
            builds = inputs.detach().cpu().numpy()
            builds = builds[0][0]
            indB = builds != 0
            im = np.zeros([256, 256, 3])
            im[:, :, 0] = (pred[0][0])
            im[:, :, 1] = (pred[0][0])
            im[:, :, 2] = (pred[0][0])
            im[indB, 2] = 100
            im = Image.fromarray(im.astype(np.uint8))

            name = str(ii)
            # file_name="buildC_carsN_simulD_DMP_samplesN/%spredict.png" %name

            file_name =   name00 + "_" + name0 + "_" + name + "DPM_predict.png"
            file_name = os.path.join(img_save_dir, file_name)
            im.save(file_name)
            # file_name="%spredict.png" %name
            Gtruth = (256 * target.detach().cpu().numpy())  # .astype(np.uint8)
            im = np.zeros([256, 256, 3])
            im[:, :, 0] = Gtruth[0][0]
            im[:, :, 1] = Gtruth[0][0]
            im[:, :, 2] = Gtruth[0][0]
            im[indB, 2] = 100
            im = Image.fromarray(im.astype(np.uint8))
            # file_name="buildC_carsN_simulD_DMP_samplesN/%starget.png" %name
            file_name =  name00 + "_" + name0 + "_" + name + "DPM_target.png"
            file_name = os.path.join(img_save_dir, file_name)
            # file_name="%starget.png" %name
            im.save(file_name)

    fig, axarr = plt.subplots(2, 2, figsize=(8, 8))
    im = os.path.join(img_save_dir, "17_187_1DPM_target.png")
    fig.add_subplot(2, 2, 1)
    plt.imshow(im)
    axarr[0, 0].set_title('Target DPM')
    im = os.path.join(img_save_dir, "17_187_1DPM_predict.png")
    fig.add_subplot(2, 2, 2)
    plt.imshow(im)
    axarr[0, 1].set_title('Predicted DPM')
    im = os.path.join(img_save_dir, "17_187_1IRT4_target.png")
    fig.add_subplot(2, 2, 3)
    plt.imshow(im)
    axarr[1, 0].set_title('Target IRT2')
    im = os.path.join(img_save_dir, "17_187_1IRT4_predict.png")
    fig.add_subplot(2, 2, 4)
    plt.imshow(im)
    axarr[1, 1].set_title('Predicted IRT2')
    fig, axarr = plt.subplots(2, 2, figsize=(8, 8))
    im = os.path.join(img_save_dir, "41_508_2DPM_target.png")
    fig.add_subplot(2, 2, 1)
    plt.imshow(im)
    axarr[0, 0].set_title('Target DPM')
    im = os.path.join(img_save_dir, "41_508_2DPM_predict.png")
    fig.add_subplot(2, 2, 2)
    plt.imshow(im)
    axarr[0, 1].set_title('Predicted DPM')
    im = os.path.join(img_save_dir, "41_508_2IRT4_target.png")
    fig.add_subplot(2, 2, 3)
    plt.imshow(im)
    axarr[1, 0].set_title('Target IRT2')
    im = os.path.join(img_save_dir, "41_508_2IRT4_predict.png")
    fig.add_subplot(2, 2, 4)
    plt.imshow(im)
    axarr[1, 1].set_title('Predicted IRT2')

    fig, axarr = plt.subplots(2, 2, figsize=(8, 8))
    im = os.path.join(img_save_dir, "31_476_2DPM_target.png")
    fig.add_subplot(2, 2, 1)
    plt.imshow(im)
    axarr[0, 0].set_title('Target DPM')
    im = os.path.join(img_save_dir, "31_476_2DPM_predict.png")
    fig.add_subplot(2, 2, 2)
    plt.imshow(im)
    axarr[0, 1].set_title('Predicted DPM')
    im = os.path.join(img_save_dir, "31_476_2IRT4_target.png")
    fig.add_subplot(2, 2, 3)
    plt.imshow(im)
    axarr[1, 0].set_title('Target IRT2')
    im = os.path.join(img_save_dir, "31_476_2IRT4_predict.png")
    fig.add_subplot(2, 2, 4)
    plt.imshow(im)
    axarr[1, 1].set_title('Predicted IRT2')