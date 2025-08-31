import torch
import torch.nn as nn
from torch.nn import functional as F








# 定义一个简单的PixelShuffle层   通道除以四
class PixelShuffle_UpSam(nn.Module):
    # 通道数会除以4
    def __init__(self, upscale_factor=2):
        super(PixelShuffle_UpSam, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(x)



# 定义一个简单的ConvTranspose2d层
class ConvTranspose_UpSam(nn.Module):
    def __init__(self, C):
        super(ConvTranspose_UpSam, self).__init__()
        # 使用反卷积进行上采样，kernel_size=4, stride=2, padding=1
        self.conv_transpose = nn.ConvTranspose2d(C, C // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv_transpose(x)

def ConvTranspose_test():

    batch_size = 4
    channels = 16
    height = 8
    width = 8
    x = torch.randn(batch_size, channels, height, width)
    print("Input shape:", x.shape)
    conv_transpose = ConvTranspose_UpSam(C=channels)
    output = conv_transpose(x)
    print("Output shape:", output.shape)

class MultiScaleUpSample(nn.Module):
    def __init__(self, in_ch, kernel_list):
        super(MultiScaleUpSample, self).__init__()
        # Initialize ModuleList for transposed convolutions
        self.t_ups = nn.ModuleList()
        for kernel_size in kernel_list:
            self.t_ups.append(
                nn.ConvTranspose2d(
                    in_ch,
                    in_ch// 2,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1
                )
            )
        # Single convolution layer after combining outputs
        self.conv = nn.Conv2d(in_ch// 2, in_ch// 2, 3, stride=1, padding=1)

    def forward(self, x, temb=None, cemb=None):
        out = None
        for t_up in self.t_ups:
            if out is None:
                out = t_up(x)
            else:
                out += t_up(x)
        out = self.conv(out)
        return out


def multiScaleUpSample_test():
    input_tensor = torch.randn(16, 16, 64, 64)
    kernel_list = [3, 5, 7]
    up_sample = MultiScaleUpSample(16, kernel_list)
    output_tensor = up_sample(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")



if __name__ == "__main__":
    # Conv3x3_DownSample_test()
    # multiScaleConvDown_test()
    # dialMultiScaleConvDown_test()
    multiScaleUpSample_test()
#

# class bilinear_UpSam(nn.Module):
#
#     def __init__(self, C):
#         super(bilinear_UpSam, self).__init__()
#         self.Up = nn.Conv2d(C, C // 2, 1, 1)
#     def forward(self, x):
#         up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         x = self.Up(up)
#         return x
#
# def bilinear_UpSam_test():
#     input_tensor = torch.randn(16, 16, 16, 16)
#     UpSam = bilinear_UpSam(16)
#     output_tensor = UpSam(input_tensor)
#     print(f"Input shape: {input_tensor.shape}")
#     print(f"Output shape: {output_tensor.shape}")
#
#
# class bilin_conv_UpSam(nn.Module):
#     def __init__(self, C):
#         super(bilin_conv_UpSam, self).__init__()
#         self.conv = nn.Conv2d(C, C // 2, kernel_size=3, stride=1, padding=1)
#     def forward(self, x):
#         up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         x = self.conv(up)
#         return x
#
# def bilin_conv_test():
#     input_tensor = torch.randn(16, 16, 16, 16)
#     UpSam = bilin_conv_UpSam(16)
#     output_tensor = UpSam(input_tensor)
#     print(f"Input shape: {input_tensor.shape}")
#     print(f"Output shape: {output_tensor.shape}")
#
# class Nearest_Conv_UpSam(nn.Module):
#     def __init__(self, C):
#         super(Nearest_Conv_UpSam, self).__init__()
#         self.conv = nn.Conv2d(C, C // 2, kernel_size=3, stride=1, padding=1)
#     def forward(self, x):
#         up = F.interpolate(x, scale_factor=2, mode='nearest')
#         x = self.conv(up)
#         return x
#
# def Nearest_Conv_test():
#     input_tensor = torch.randn(16, 16, 16, 16)
#     UpSam = Nearest_Conv_UpSam(16)
#     output_tensor = UpSam(input_tensor)
#     print(f"Input shape: {input_tensor.shape}")
#     print(f"Output shape: {output_tensor.shape}")

# class Avg_DownSampling2D(nn.Module):
#     def __init__(self, C):
#         super(Avg_DownSampling2D, self).__init__()
#         self.Down = nn.Sequential(
#             # 使用平均池化进行下采样，stride=2表示宽高各减半
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(C),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x):
#         return self.Down(x)
#
# def Avg_Down_test():
#     input_tensor = torch.randn(16, 16, 128, 128)
#     DownSamp_exm = Avg_DownSampling2D(16)
#     output_tensor = DownSamp_exm(input_tensor)
#     print(f"Input shape: {input_tensor.shape}")
#     print(f"Output shape: {output_tensor.shape}")