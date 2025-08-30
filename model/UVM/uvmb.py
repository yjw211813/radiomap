import torch 
import torch.nn as nn
from mamba_ssm import Mamba
import torch.cuda as cuda
class UVMB(nn.Module):
    def __init__(self,c=3,w=256,h=256):
        super().__init__()
        self.convb  = nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=16, out_channels=c, kernel_size=3, stride=1, padding=1)
                        )
        self.model1 = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.model3 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=w*h, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.softmax = nn.Softmax()
    def forward(self, x):
        b,c,w,h = x.shape
        x = self.convb(x) + x
        x = self.ln(x.reshape(b, -1, c))
        y = self.model1(x).permute(0, 2, 1)
        z = self.model3(y).permute(0, 2, 1)
        att = self.softmax(self.model2(x))
        result = att * z
        output = result.reshape(b, c, w, h)
        return self.smooth(output)


def test_UVMB():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 1. 初始化模型（自动并行化处理）
    c, w, h = 3, 128,128
    print(f"初始化前显存: {cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    model = UVMB(c=3, w=w, h=h)  # 先在CPU创建
    model = model.to(device)  # 再移到GPU
    print(f"初始化后显存: {cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    # 2. 输入数据生成（确保requires_grad一致性）
    x = torch.randn(1, c, w, h, requires_grad=True).to(device)
    print(x.shape)
    output = model(x)
    print(output.shape)





if __name__ == "__main__":
    test_UVMB()


