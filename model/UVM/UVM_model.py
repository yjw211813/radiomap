""" Full assembly of the parts to form the complete network """

from model.UVM.unet_part import *

from model.sub_block.statistic_tools import gpu_statistic
class UVMNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UVMNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

    def forward(self, inp):
        x = inp
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x + inp[:,-2,:,:].unsqueeze(1)


def test_UNet():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    get_gpu_info = gpu_statistic(device)
    c, w, h = 2, 256,256
    model =UVMNet(n_channels=2)  # 先在CPU创建
    x = torch.randn(16, c, w, h, requires_grad=True)

    get_gpu_info.print_gpu_memory("UVMB GPU info", x, model)

if __name__ == "__main__":
    test_UNet()