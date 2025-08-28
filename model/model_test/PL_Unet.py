from pydoc import importfile
import sys
import os
import h5py
from torch.utils.data import Dataset, DataLoader

# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.sub_block.mid_conv import *
from environment_code.tif_convert_height import map_info


#### 或者采用元素重排上采样

class map_meas_UNet(nn.Module):
    def __init__(self):
        super(map_meas_UNet, self).__init__()
        C_out1 = 16
        C_out2 = 32
        C_out3 = 64
        C_out4 = 128
        C_out5 = 256
        self.downconv1 =  Inception_ghost2D(C_in=2, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down1 = Conv_DownSampling2D(C_out1)   #  下采样 (100,C_out1,240,240) 通道不变下采样
        self.downconv2 =   Inception_ghost2D(C_in=C_out1, C_out=C_out2, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down2 = Conv_DownSampling2D(C_out2)   #  下采样 (100,C_out2,120,120)
        self.downconv3 =    Inception_ghost2D(C_in=C_out2, C_out=C_out3, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down3 = Conv_DownSampling2D(C_out3)  #  下采样 (100,C_out3,60,60)
        self.downconv4 =   Inception_ghost2D(C_in=C_out3, C_out=C_out4, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down4 = Conv_DownSampling2D(C_out4)  #  下采样 (100,C_out4,30,30)
        self.downconv5 =   Fractal_inception2D(input_channel=C_out4, output_channel=C_out5)
        self.down5 = Conv_DownSampling2D(C_out5)  #  下采样 (100,C_out5,15,15)

        self.conv_center =   Fractal_inception2D(input_channel=C_out5, output_channel=C_out5)

        self.upsam5 = bilin_conv_UpSam(C_out5) #  下采样 (100,C_out4,30,30)
        self.upconv5 = Fractal_inception2D(input_channel=C_out4+C_out4, output_channel=C_out4)
        self.upsam4 = bilin_conv_UpSam(C_out4)#  下采样 (100,C_out3,60,60)
        self.upconv4 = Inception_ghost2D(C_in=C_out3+C_out3, C_out=C_out3, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam3 = bilin_conv_UpSam(C_out3)#  下采样 (100,C_out2,120,120)
        self.upconv3 = Inception_ghost2D(C_in=C_out2+C_out2, C_out=C_out2, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam2 = bilin_conv_UpSam(C_out2)#  下采样 (100,C_out1,240,240)
        self.upconv2 = Inception_ghost2D(C_in=C_out1+C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam1 = bilin_conv_UpSam(C_out1)  # 下采样 (100,C_out1/2,480,480)
        self.pred = torch.nn.Conv2d(int(C_out1/2), 1, 3, 1, 1)

    def forward(self, input):
        # 下采样路径
        down1_out = self.down1(self.downconv1(input))
        down2_out = self.down2(self.downconv2(down1_out))
        down3_out = self.down3(self.downconv3(down2_out))
        down4_out = self.down4(self.downconv4(down3_out))
        center_out = self.conv_center(self.down5(self.downconv5(down4_out)))
        # 上采样路径
        up5_out = self.upsam5(center_out)
        up5_out = self.upconv5(torch.cat([up5_out, down4_out], dim=1))  # 拼接
        up4_out = self.upsam4(up5_out)
        up4_out = self.upconv4(torch.cat([up4_out, down3_out], dim=1))  # 拼接
        up3_out = self.upsam3(up4_out)
        up3_out = self.upconv3(torch.cat([up3_out, down2_out], dim=1))  # 拼接
        up2_out = self.upsam2(up3_out)
        up2_out = self.upconv2(torch.cat([up2_out, down1_out], dim=1))  # 拼接
        out =  self.pred(self.upsam1(up2_out))
        return out

def map_meas_UNet_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = map_meas_UNet().to(device)

    # 创建一个输入张量 (2, 2, 480, 480)
    input_tensor = torch.randn(8, 2, 480, 480).to(device)
    # 通过模型进行前向传播
    output = model(input_tensor)

    print(f"Output shape: {output.shape}")


class BTM_Net_v2(nn.Module):
    def __init__(self,img_H,img_W,feature_num):
        super(BTM_Net_v2, self).__init__()
        C_list = [16,16,16,16,32,32,32,32,64,64,64,64]
        self.hidden_mul = 20
        self.feature_num = feature_num
        self.img_H = img_H
        self.img_W = img_W
        self.feature_num = feature_num
        gcd_result = self.get_gcd(img_H, img_W)
        down_samp_time = int(math.floor(math.log2(gcd_result/30))) + 1
        # 构建编码器
        layers = []
        layers.append(Inception_ghost2D(C_in=1, C_out=C_list[0], kernel_sizes=[1, 3, 5, 7], dilated_num=1))
        for i in range(down_samp_time):
            layers.append(Conv_DownSampling2D(C_list[i]))  # 添加下采样层
            layers.append(
                Inception_ghost2D(C_in=C_list[i], C_out=C_list[i + 1], kernel_sizes=[1, 3, 5, 7], dilated_num=1))
        self.encoder = nn.Sequential(*layers)
        self.compress = torch.nn.Conv2d(C_list[down_samp_time], 1, 3, 1, 1)
        out_H = img_H//(2**down_samp_time)
        out_W = img_W // (2 ** down_samp_time)
        self.fc_weight = nn.Linear(out_H*out_W, self.feature_num*self.feature_num*self.hidden_mul)
        self.fc_map = nn.Linear(self.feature_num*self.hidden_mul, img_H * img_W)
        self.gelu = nn.GELU()

    def forward(self, stalite_map,site_info):
        batch_size = site_info.shape[0]
        out_weight = self.fc_weight(self.compress(self.encoder(stalite_map)).reshape(batch_size, -1))
        out_weight = out_weight.reshape(batch_size, self.feature_num, self.feature_num*self.hidden_mul)
        feature_in = site_info.squeeze(1)
        feature_out = torch.matmul(feature_in, out_weight)
        feature_out = self.gelu(self.fc_map(feature_out)).squeeze(1)
        out_map = feature_out.reshape(batch_size, self.img_H, self.img_W).unsqueeze(1)
        return out_map

    def get_gcd(self,a, b):
        while b != 0:
            a, b = b, a % b
        return a

def BTM_Net_v2_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_num = 4
    batch_size = 8
    img_H = 480
    img_W =480
    site_info = torch.randn(batch_size, 1, 1, feature_num).to(device)
    input_map = torch.randn(batch_size, 1, img_H, img_W).to(device)
    net = BTM_Net_v2(img_H,img_W,feature_num).to(device)
    output = net(input_map,site_info)
    print(f"Output shape: {output.shape}")





class map_meas_pos_UNet(nn.Module):
    def __init__(self,img_H,img_W,feature_num):
        super(map_meas_pos_UNet, self).__init__()
        C_out1 = 16
        C_out2 = 32
        C_out3 = 64
        C_out4 = 128
        C_out5 = 256
        self.downconv1 =   Inception_ghost2D(C_in=2, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down1 = Conv_DownSampling2D(C_out1)  #  下采样 (100,C_out1,240,240)
        self.downconv2 =    Inception_ghost2D(C_in=C_out1, C_out=C_out2, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down2 = Conv_DownSampling2D(C_out2)  #  下采样 (100,C_out2,120,120)
        self.downconv3 =   Inception_ghost2D(C_in=C_out2, C_out=C_out3, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down3 = Conv_DownSampling2D(C_out3)  #  下采样 (100,C_out3,60,60)
        self.downconv4 =   Inception_ghost2D(C_in=C_out3, C_out=C_out4, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down4 = Conv_DownSampling2D(C_out4)  #  下采样 (100,C_out4,30,30)
        self.downconv5 =   Inception_ghost2D(C_in=C_out4, C_out=C_out5, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down5 = Conv_DownSampling2D(C_out5)  #  下采样 (100,C_out4,15,15)

        self.conv_center =   Fractal_inception2D(input_channel=C_out5, output_channel=C_out5)

        self.upsam5 = PixelShuffle_UpSam() #  下采样 (100,C_out4,30,30)
        self.upconv5 = Inception_ghost2D(C_in=C_out4+C_out4, C_out=C_out4+C_out4, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam4 = PixelShuffle_UpSam()#  下采样 (100,C_out3,60,60)
        self.upconv4 = Inception_ghost2D(C_in=C_out3+C_out3, C_out=C_out3+C_out3, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam3 = PixelShuffle_UpSam()#  下采样 (100,C_out2,120,120)
        self.upconv3 = Inception_ghost2D(C_in=C_out2+C_out2, C_out=C_out2+C_out2, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam2 = PixelShuffle_UpSam()#  下采样 (100,C_out1,240,240)
        self.upconv2 = Inception_ghost2D(C_in=C_out1+C_out1, C_out=C_out1+C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.upsam1 = PixelShuffle_UpSam()  # 下采样 (100,C_out1/2,480,480)
        self.atten1 = BTM_Net_v2( img_H, img_W,feature_num)
        self.pred = torch.nn.Conv2d(int(C_out1/2), 1, 3, 1, 1)

    def forward(self, input_data,site_info):
        geo_map_1 = input_data[:,0,:,:].unsqueeze(1)
        # 下采样路经
        down1_out = self.down1(self.downconv1(input_data))
        down2_out = self.down2(self.downconv2(down1_out))
        down3_out = self.down3(self.downconv3(down2_out))
        down4_out = self.down4(self.downconv4(down3_out))
        down5_out = self.down5(self.downconv5(down4_out))

        ### 卷积 上采样 注意力 拼接 卷积 上采样 ....
        center_out = self.conv_center(down5_out)
        up5_out = self.upsam5(torch.cat([center_out, down5_out], dim=1))  #  下采样 (100,C_out4,30,30)

        up4_out = self.upconv5(torch.cat([up5_out, down4_out], dim=1))  # 拼接
        up4_out = self.upsam4(up4_out) #  下采样 (100,C_out3,60,60)

        up3_out = self.upconv4(torch.cat([up4_out , down3_out], dim=1))  # 拼接
        up3_out = self.upsam3(up3_out) #  下采样 (100,C_out3,60,60)

        up2_out = self.upconv3(torch.cat([up3_out, down2_out], dim=1))  # 拼接
        up2_out = self.upsam2(up2_out) #  下采样 (100,C_out3,60,60)

        up1_out = self.upconv2(torch.cat([up2_out, down1_out], dim=1))  # 拼接
        up1_out = self.upsam1(up1_out)
        atten1_out = self.atten1(geo_map_1, site_info).expand(-1, up1_out.shape[1], -1, -1)
        out =  self.pred(up1_out * atten1_out)

        return out

def map_meas_pos_UNet_test():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 8
    feature_num = 4
    img_H = 480
    img_W = 480
    site_info = torch.randn(batch_size, 1, 1, feature_num).to(device)
    input_map = torch.randn(batch_size, 2, img_H, img_W).to(device)
    net = map_meas_pos_UNet(img_H, img_W,feature_num).to(device)
    output = net(input_map, site_info)
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':

    # map_meas_UNet_test()
    map_meas_pos_UNet_test()
    # BTM_Net_v2_test()
