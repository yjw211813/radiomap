import math
from pydoc import importfile
import sys
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.sub_block.mid_conv import *
from environment_code.generate_radio_map_data import source_feature,data_structure
from environment_code.tif_convert_height import map_info


class location_model(nn.Module):
    def __init__(self,img_H,img_W,out_dim):
        super(location_model, self).__init__()
        C_out1 = 16
        C_out2 = 16
        C_out3 = 32
        C_out4 = 32
        C_out5 = 64
        down_samp_time = 5
        out_H = img_H//(2**down_samp_time)
        out_W = img_W // (2 ** down_samp_time)
        middle_layer = 128
        self.downconv1 =  Inception_ghost2D(C_in=2, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down1 = Conv_DownSampling2D(C_out1)   #  下采样 (8,C_out1,240,240) 通道不变下采样
        self.downconv2 =   Inception_ghost2D(C_in=C_out1, C_out=C_out2, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down2 = Conv_DownSampling2D(C_out2)   #  下采样 (8,C_out2,120,120)
        self.downconv3 =    Inception_ghost2D(C_in=C_out2, C_out=C_out3, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down3 = Conv_DownSampling2D(C_out3)  #  下采样 (8,C_out3,60,60)
        self.downconv4 =   Inception_ghost2D(C_in=C_out3, C_out=C_out4, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down4 = Conv_DownSampling2D(C_out4)  #  下采样 (8,C_out4,30,30)
        self.downconv5 =   Fractal_inception2D(input_channel=C_out4, output_channel=C_out5)
        self.down5 = Conv_DownSampling2D(C_out5)  #  下采样 (8,C_out5,15,15)
        self.conv_center =   Fractal_inception2D(input_channel=C_out5, output_channel=C_out5)

        self.fc_mid = nn.Linear(C_out5 * out_H * out_W, middle_layer)
        self.fc_out = nn.Linear(middle_layer, out_dim)
        self.gelu = nn.GELU()


    def forward(self, input):
        batch_size = input.shape[0]
        # 下采样路径
        down1_out = self.down1(self.downconv1(input))
        down2_out = self.down2(self.downconv2(down1_out))
        down3_out = self.down3(self.downconv3(down2_out))
        down4_out = self.down4(self.downconv4(down3_out))
        center_out = self.conv_center(self.down5(self.downconv5(down4_out)))
        out =  self.fc_out(self.gelu(self.fc_mid(center_out.reshape(batch_size, -1))))
        return out

def map_meas_UNet_test():
    img_H = 480
    img_W = 480
    out_dim = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = location_model(img_H,img_W,out_dim).to(device)

    # 创建一个输入张量 (2, 2, 480, 480)
    input_tensor = torch.randn(8, 2, 480, 480).to(device)
    # 通过模型进行前向传播
    output = model(input_tensor)

    print(f"Output shape: {output.shape}")

if __name__ == '__main__':
    map_meas_UNet_test()