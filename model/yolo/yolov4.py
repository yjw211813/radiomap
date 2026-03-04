import torch
import torch.nn as nn

from model.yolo.conv import Conv, ConvTranspose
from model.yolo.block import C3k2, SPPF, C2PSA



class Yolov4(nn.Module):
    """
    Task:
        Input : [B, 2, 256, 256]
            - channel 0: obstacle map
            - channel 1: free-space path loss from single light source
        Output: [B, 1, 256, 256]
            - generated illumination map
    """

    def __init__(self, in_channels=2, base_ch=64, *args, **kargs):
        super().__init__(*args, **kargs)

        # =========================
        # Encoder (slow downsampling)
        # =========================
        self.stem = Conv(in_channels, base_ch, k=3, s=1)        # 256

        self.down1 = Conv(base_ch, base_ch * 2, k=3, s=2)       # 128
        self.enc1 = C3k2(base_ch * 2, base_ch * 2, n=2)

        self.down2 = Conv(base_ch * 2, base_ch * 4, k=3, s=2)   # 64
        self.enc2 = C3k2(base_ch * 4, base_ch * 4, n=2)

        self.down3 = Conv(base_ch * 4, base_ch * 8, k=3, s=2)   # 32
        self.enc3 = C3k2(base_ch * 8, base_ch * 8, n=2, shortcut=True)

        # =========================
        # Bottleneck (global context)
        # =========================
        self.bottleneck = nn.Sequential(
            SPPF(base_ch * 8, base_ch * 8, k=5),
            C2PSA(base_ch * 8, base_ch * 8)
        )

        # =========================
        # Decoder (strictly symmetric)
        # =========================
        self.up3 = ConvTranspose(base_ch * 8, base_ch * 4)     # 32 → 64
        self.dec3 = C3k2(base_ch * 8, base_ch * 4, n=2)

        self.up2 = ConvTranspose(base_ch * 4, base_ch * 2)     # 64 → 128
        self.dec2 = C3k2(base_ch * 4, base_ch * 2, n=2)

        self.up1 = ConvTranspose(base_ch * 2, base_ch)         # 128 → 256
        self.dec1 = C3k2(base_ch * 2, base_ch, n=2)

        # =========================
        # Output head
        # =========================
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # -------- Encoder --------
        x0 = self.stem(x)            # 256
        x1 = self.enc1(self.down1(x0))  # 128
        x2 = self.enc2(self.down2(x1))  # 64
        x3 = self.enc3(self.down3(x2))  # 32

        # -------- Bottleneck --------
        xb = self.bottleneck(x3)

        # -------- Decoder --------
        y3 = self.up3(xb)
        y3 = self.dec3(torch.cat([y3, x2], dim=1))

        y2 = self.up2(y3)
        y2 = self.dec2(torch.cat([y2, x1], dim=1))

        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, x0], dim=1))

        return self.sigmoid(self.out_conv(y1))


if __name__ == '__main__':
    from torchsummary import summary
    model = Yolov4()
    summary(model.cuda(), input_size=(2, 256, 256), batch_size=2)
