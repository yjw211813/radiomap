import torch
import torch.nn as nn
from model.conv2D_block import Inception_ghost2D,Conv_DownSampling2D,Fractal_inception2D


# 改进型UNet 需要思考其中参数量

class formula_map(nn.Module):
    def __init__(self,feature_num,img_H,img_W):
        super(formula_map, self).__init__()
        # 输入480*480
        # 输入参数定义
        C_out1 = 32
        down_num = 5
        self.hidden_mul1 = 10
        self.hidden_mul2 = 20
        self.hidden_mul_bias = 20
        self.feature_num = feature_num
        self.img_H = img_H
        self.img_W = img_W

        self.encoder = nn.Sequential(
            Inception_ghost2D(C_in=1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),   #  下采样 (100,C_out1,240,240)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),   #  下采样 (100,C_out1,120,120)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),  #  下采样 (100,C_out1,60,60)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),  #  下采样 (100,C_out1,30,30)
            Fractal_inception2D(input_channel=C_out1, output_channel=C_out1),
            Conv_DownSampling2D(C_out1)  #  下采样 (100,C_out1,15,15)
        )
        out_H = int(img_H / (2 ** down_num))
        out_W = int(img_W / (2 ** down_num))

        self.fc_weight = nn.Linear(C_out1 * out_H * out_W, feature_num*self.hidden_mul1 * feature_num*self.hidden_mul2)

        self.fc1 = nn.Linear(feature_num, feature_num*self.hidden_mul1)
        self.fc2 = nn.Linear(feature_num*self.hidden_mul1, feature_num*self.hidden_mul2)
        self.fc_map =nn.Linear(feature_num*self.hidden_mul2, img_H*img_W)

        self.fc_bias1 = nn.Linear(feature_num, feature_num*self.hidden_mul_bias)
        self.fc_bias2 = nn.Linear(feature_num*self.hidden_mul_bias, 1)
        self.gelu = nn.GELU()

    def forward(self, samp_map,Hata_map,feature_in):
        feature_in = feature_in.squeeze(1)
        batch_size = samp_map.shape[0]
        out_weight = self.fc_weight(self.encoder(samp_map).reshape(batch_size,-1))
        out_weight = out_weight.reshape(batch_size, self.feature_num*self.hidden_mul1, self.feature_num * self.hidden_mul2)
        feature_out = self.gelu(self.fc1(feature_in))

        feature_out = torch.matmul(feature_out, out_weight).squeeze(1)
        out_map = self.gelu(self.fc_map(feature_out))
        bias = self.fc_bias2(self.gelu(self.fc_bias1(feature_in))).unsqueeze(1)
        out_map =  out_map.reshape(batch_size,1,Hata_map.shape[2],Hata_map.shape[3])
        out_map = out_map + Hata_map + bias

        return out_map
# 模型测试程序
def formula_map_forward_test():
    feature_num = 4
    img_H = 480
    img_W = 480
    feature_data = torch.rand(5,1,1,feature_num)
    img_data = torch.rand(5, 1, img_H, img_W)
    Hata_map = torch.rand(5, 1, img_H, img_W)
    net = formula_map(feature_num,img_H,img_W)
    torch.save(net.state_dict() , f"../runs/model_pth/checkpoint_epoch.pth")
    output = net(img_data,Hata_map,feature_data)
    print("output shape :",output.shape)


if __name__ == '__main__':
    formula_map_forward_test()