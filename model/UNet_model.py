import torch
import sys
import os
# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.conv2D_block import *

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
# 变换到相同形状进行加和形式

class BTM_Net_v3(nn.Module):
    def __init__(self,input_shape, output_shape,C_list):
        super(BTM_Net_v3, self).__init__()
        self.input_channel , self.input_H , self.input_W = input_shape
        self.output_channel , self.output_H , self.output_W = output_shape

        C_list = torch.cat((C_list, torch.tensor([self.output_channel], dtype=torch.int32)))
        # 构建编码器
        layers = []
        layers.append(Inception_ghost2D(C_in=self.input_channel, C_out=C_list[0], kernel_sizes=[1, 3, 5, 7], dilated_num=1))
        for i in range(len(C_list) - 1):
            layers.append(Res_Inception_ghost2D(C_in=C_list[i], C_out=C_list[i + 1], kernel_sizes=[1, 3, 5, 7], dilated_num=1))
        self.encoder = nn.Sequential(*layers)
        self.gelu = nn.GELU()

    def forward(self, condition_map):

        resized_map = F.interpolate(condition_map, size=(self.output_H, self.output_W), mode='bilinear', align_corners=True)
        out_map = self.encoder(resized_map)

        return out_map


def BTM_Net_v3_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    input_shape = [4,256,256]
    output_shape = [64,64,64]
    C_list = [64,64,64,128,128,128,128]
    condition_map = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]).to(device)

    net = BTM_Net_v3(input_shape,output_shape,C_list).to(device)
    output = net(condition_map)
    print(f"Output shape: {output.shape}")



class BTM_ghost_UNet(nn.Module):
    def __init__(self,input_shape,output_shape,C_down_list,C_list_attn):
        super(BTM_ghost_UNet, self).__init__()

        self.input_channel, self.input_H, self.input_W = input_shape
        self.output_channel, self.output_H, self.output_W = output_shape

        self.downconv1 =   Inception_ghost2D(C_in=self.input_channel, C_out=C_down_list[0], kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down1 = Conv_DownSampling2D(C_down_list[0])  #  下采样 (100,C_out0,128,128)
        self.downconv2 =    Inception_ghost2D(C_in=C_down_list[0], C_out=C_down_list[1], kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down2 = Conv_DownSampling2D(C_down_list[1])  #  下采样 (100,C_out1,64,64)
        self.downconv3 =   Inception_ghost2D(C_in=C_down_list[1], C_out=C_down_list[2], kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down3 = Conv_DownSampling2D(C_down_list[2])  #  下采样 (100,C_out2,32,32)
        self.downconv4 =   Inception_ghost2D(C_in=C_down_list[2], C_out=C_down_list[3], kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down4 = Conv_DownSampling2D(C_down_list[3])  #  下采样 (100,C_out3,16,16)

        self.conv_center = Res_Fractal_inception2D(input_channel=C_down_list[3], output_channel=C_down_list[3])

        self.upconv4 = Inception_ghost2D(C_in=C_down_list[3]+C_down_list[3], C_out=C_down_list[3]+C_down_list[3], kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam4 = PixelShuffle_UpSam()  # 下采样 (100, C_out3/2 ,32,32)
        # 记得repeat
        BTM_atten4_input_shape = [self.input_channel, self.input_H, self.input_W]
        BTM_atten4_output_shape = [C_down_list[2]//4, self.input_H // 8, self.input_W // 8]
        self.BTM_atten4 = BTM_Net_v3(BTM_atten4_input_shape, BTM_atten4_output_shape, C_list_attn)


        self.upconv3 = Inception_ghost2D(C_in=C_down_list[2]+C_down_list[2], C_out=C_down_list[2]+C_down_list[2], kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam3 = PixelShuffle_UpSam()#  下采样 (100, C_out2/2 ,64,64)
        # 记得repeat
        BTM_atten3_input_shape = [self.input_channel, self.input_H, self.input_W]
        BTM_atten3_output_shape = [C_down_list[1]//4, self.input_H // 4, self.input_W // 4]
        self.BTM_atten3 = BTM_Net_v3(BTM_atten3_input_shape, BTM_atten3_output_shape, C_list_attn // 2)


        self.upconv2 = Inception_ghost2D(C_in=C_down_list[1]+C_down_list[1], C_out=C_down_list[1]+C_down_list[1], kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam2 = PixelShuffle_UpSam()#  下采样 (100, C_out1/2 ,128,128)
        BTM_atten2_input_shape = [self.input_channel, self.input_H, self.input_W]
        BTM_atten2_output_shape = [C_down_list[0]//4, self.input_H // 2, self.input_W // 2]
        self.BTM_atten2 = BTM_Net_v3(BTM_atten2_input_shape, BTM_atten2_output_shape, C_list_attn // 4)


        self.upconv1 = Inception_ghost2D(C_in=C_down_list[0]+C_down_list[0], C_out=C_down_list[0]+C_down_list[0], kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam1 = PixelShuffle_UpSam()  # 下采样 (100,C_out0/2,256,256)
        BTM_atten1_input_shape = [self.input_channel, self.input_H, self.input_W]
        BTM_atten1_output_shape = [C_down_list[0]//4, self.input_H , self.input_W ]
        self.BTM_atten1 = BTM_Net_v3(BTM_atten1_input_shape, BTM_atten1_output_shape, C_list_attn // 4)


        now_ch = C_down_list[0]//2
        self.tail = nn.Sequential(
            nn.GroupNorm(now_ch//4, now_ch),
            Swish(),
            nn.Conv2d(now_ch, self.output_channel, 3, stride=1, padding=1)
        )

    def forward(self, input_data):
        # 下采样路经
        down1_out = self.down1(self.downconv1(input_data))
        down2_out = self.down2(self.downconv2(down1_out))
        down3_out = self.down3(self.downconv3(down2_out))
        down4_out = self.down4(self.downconv4(down3_out))

        ### 卷积 上采样 注意力 拼接 卷积 上采样 ....
        up4_out = self.upsam4(self.upconv4(torch.cat([self.conv_center(down4_out), down4_out], dim=1)))  # 拼接
        atten4_out = self.BTM_atten4(input_data).repeat(1, 4, 1, 1)

        up3_out = self.upsam3(self.upconv3(torch.cat([up4_out * atten4_out, down3_out], dim=1)))  # 拼接
        atten3_out = self.BTM_atten3(input_data).repeat(1, 4, 1, 1)

        up2_out = self.upsam2(self.upconv2(torch.cat([up3_out * atten3_out, down2_out], dim=1)))  # 拼接
        atten2_out = self.BTM_atten2(input_data).repeat(1, 4, 1, 1)

        up1_out = self.upsam1(self.upconv1(torch.cat([up2_out * atten2_out, down1_out], dim=1)))  # 拼接
        atten1_out = self.BTM_atten1(input_data).repeat(1, 2, 1, 1)

        out =  self.tail(up1_out * atten1_out)

        return out

    def load_weights(self, checkpoint_path):
        """加载预训练权重"""
        self.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded weights from {checkpoint_path}")



def BTM_ghost_UNet_test():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 8
    feature_num = 4
    img_H = 256
    img_W = 256
    input_data = torch.randn(batch_size, 4, img_H, img_W).to(device)

    BTM_ghost_UNet_input_shape = [input_data.shape[1], input_data.shape[2], input_data.shape[3]]
    BTM_ghost_UNet_output_shape = [1, input_data.shape[2], input_data.shape[3]]
    C_down_list = [32, 64, 128, 256]
    C_list_attn = torch.tensor([64, 64, 64, 128, 128, 128, 128])
    net = BTM_ghost_UNet(BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape,C_down_list,C_list_attn).to(device)
    output = net(input_data)
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':


    # map_meas_pos_UNet_test()
    BTM_ghost_UNet_test()
