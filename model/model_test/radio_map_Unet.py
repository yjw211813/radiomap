import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
import sys
import os
# 将上一级目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
class InceptionModule(nn.Module):
    def __init__(self, C_in, C_out, drop_out):
        super(InceptionModule, self).__init__()

        if C_out % 4 != 0:
            raise ValueError(f"C_out ({C_out}) must be divisible by 4.")

        sub_Cout = int(C_out / 4)
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(sub_Cout),
            nn.Dropout(drop_out),
            nn.LeakyReLU()
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(sub_Cout),
            nn.Dropout(drop_out),
            nn.LeakyReLU()
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(sub_Cout),
            nn.Dropout(drop_out),
            nn.LeakyReLU()
        )

        self.branch7x7 = nn.Sequential(
            nn.Conv2d(C_in, sub_Cout, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(sub_Cout),
            nn.Dropout(drop_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch3 = self.branch3x3(x)
        branch5 = self.branch5x5(x)
        branch7 = self.branch7x7(x)

        # 拼接所有分支的输出
        outputs = [branch1, branch3, branch5, branch7]
        return torch.cat(outputs, 1)  # 在通道维度上拼接
# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.inception1 = InceptionModule(C_in, C_out,0.2)
        self.inception2 = InceptionModule(C_out, C_out,0.2)

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        return x
# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)

class BTM_Net(nn.Module):
    def __init__(self, feature_num, img_H, img_W):
        super(BTM_Net, self).__init__()
        # 4次conv
        self.C1 = Conv(1, 4)
        self.C2 = Conv(4, 8)
        self.C3 = Conv(8, 4)
        self.pred = torch.nn.Conv2d(4, 1, 3, 1, 1)
        self.Th = torch.nn.Sigmoid()
        self.fc1 = nn.Linear(feature_num, img_H * img_W)
        self.fc2 = nn.Linear(feature_num, img_H * img_W)
        self.fc3 = nn.Linear(feature_num, img_H * img_W)
        self.fc4 = nn.Linear(feature_num, img_H * img_W)

    def forward(self, x, feature):

        batch_size, _, img_H, img_W = x.shape  # 从x的形状获取高度和宽度

        #################################
        #################################
        R1 = self.C1(x)
        FR1 = self.fc1(feature).reshape(R1.size(0), -1, img_H, img_W)
        Y1 = R1 * FR1.expand(-1, 4, -1, -1)  # 点乘
        R2 = self.C2(Y1)
        FR2 = self.fc2(feature).reshape(R2.size(0), -1, img_H, img_W)
        Y2 = R2 * FR2.expand(-1, 8, -1, -1)  # 点乘
        R3 = self.C3(Y2)
        FR3 = self.fc3(feature).reshape(R3.size(0), -1, img_H, img_W)
        Y3 = R3 * FR3.expand(-1, 4, -1, -1)  # 点乘
        R4 = self.pred(Y3)
        FR4 = self.fc4(feature).reshape(R4.size(0), -1, img_H, img_W)
        Y4 = self.Th(R4 * FR4.expand(-1, 1, -1, -1))  # 点乘

        return Y4
# 主干网络
class UNet(nn.Module):

    def __init__(self,img_C,img_H,img_W,feature_num):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(img_C, 16)
        self.D1 = DownSampling(16)
        self.C2 = Conv(16, 32)
        self.D2 = DownSampling(32)
        self.C3 = Conv(32, 64)
        self.D3 = DownSampling(64)
        self.C4 = Conv(64, 128)
        self.D4 = DownSampling(128)
        self.C5 = Conv(128, 256)

        # 4次上采样
        self.U1 = UpSampling(256)
        self.atten1 = BTM_Net(feature_num, img_H // 8, img_W // 8)
        self.C6 = Conv(256, 128)
        self.U2 = UpSampling(128)
        self.atten2 = BTM_Net(feature_num, img_H // 4, img_W // 4)
        self.C7 = Conv(128, 64)
        self.U3 = UpSampling(64)
        self.atten3 = BTM_Net(feature_num, img_H // 2, img_W // 2)
        self.C8 = Conv(64, 32)
        self.U4 = UpSampling(32)
        self.atten4 = BTM_Net(feature_num, img_H , img_W )
        self.C9 = Conv(32, 16)
        self.pred = torch.nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x1,feature_array):


        # 下采样部分
        R1 = self.C1(x1)
        # 提取 x1[:, 0, :, :] 并扩展维度以适应插值的输入格式
        # 变为 (4, 1, 320, 240)
        x1_map = x1[:, 0, :, :].unsqueeze(1)  # 变为 (4, 1, 320, 240)
        # 使用插值方法进行缩放，得到 x2_map，目标形状为 (4, 1, 160, 120)
        x2_map = F.interpolate(x1_map, size=(x1_map.shape[2] // 2, x1_map.shape[3] // 2), mode='bilinear',align_corners=False)
        # 再次进行插值，得到 x3_map，目标形状为 (4, 1, 80, 60)
        x3_map = F.interpolate(x2_map, size=(x2_map.shape[2] // 2, x2_map.shape[3] // 2), mode='bilinear',align_corners=False)
        # 再次进行插值，得到 x4_map，目标形状为 (4, 1, 40, 30)
        x4_map = F.interpolate(x3_map, size=(x3_map.shape[2] // 2, x3_map.shape[3] // 2), mode='bilinear',align_corners=False)

        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        ############################################
        #################################

        O1 = self.C6(self.atten1(x4_map, feature_array).expand(-1, 256, -1, -1) * self.U1(Y1, R4))
        O2 = self.C7(self.atten2(x3_map, feature_array).expand(-1, 128, -1, -1) * self.U2(O1, R3))
        O3 = self.C8(self.atten3(x2_map, feature_array).expand(-1, 64, -1, -1) * self.U3(O2, R2))
        O4 = self.C9(self.atten4(x1_map, feature_array).expand(-1, 32, -1, -1) * self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.pred(O4)

def BTM_Net_test():
    img_data = torch.randn(10,1,240,240)
    feature_data = torch.randn(10, 1, 1, 5)
    BTM_Net1 = BTM_Net(feature_num=5, img_H=img_data.shape[2], img_W=img_data.shape[3])
    output_tensor = BTM_Net1(img_data, feature_data)
    # 打印输出形状
    print(f"BTM_Net Input shape: {img_data.shape}")
    print(f"BTM_Net Output shape: {output_tensor.shape}")

def UNet_test():
    num_batch = 16
    img_C, img_H, img_W, feature_num = 2, 480, 480,4
    input_img = torch.rand(num_batch, img_C, img_H, img_W)
    original_feature = torch.randn(num_batch, 1, 1, feature_num)

    test_net = UNet(img_C, img_H, img_W, feature_num)
    output_tensor = test_net(input_img, original_feature)
    # 打印输出形状
    print(f"UNet input_img shape: {input_img.shape}")
    print(f"UNet Output shape: {output_tensor.shape}")



if __name__ == "__main__":
    BTM_Net_test()
    UNet_test()