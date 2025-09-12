import torch
import torch.nn as nn

# 假设 `one_hot_labelsF` 和 `one_hot_labelsR` 是通过 ohe_vector_from_labels 函数生成的
def ohe_vector_from_labels(labels, num_classes):
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# 假设当前设备是 CPU 或 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 假设 batch_sz 为 4，num_classes 为 2，输入图像的尺寸是 (batch_sz, 3, 32, 32)
batch_sz = 4
num_classes = 2
inps = torch.randn(batch_sz, 3, 32, 32).to(device)  # 模拟一个 batch 的图像输入

# 生成 one-hot 标签
one_hot_labelsF = ohe_vector_from_labels(torch.tensor([0] * batch_sz).to(device), num_classes)
one_hot_labelsR = ohe_vector_from_labels(torch.tensor([1] * batch_sz).to(device), num_classes)

# 将标签扩展到四维 (batch_sz, num_classes, 1, 1)
image_one_hot_labelsF = one_hot_labelsF[:, :, None, None]
image_one_hot_labelsR = one_hot_labelsR[:, :, None, None]

# 输出标签的尺寸
print("Before repeat:")
print("image_one_hot_labelsF shape:", image_one_hot_labelsF.shape)
print("image_one_hot_labelsR shape:", image_one_hot_labelsR.shape)

# 模拟网络和优化器
class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)  # 一个简单的卷积层

    def forward(self, x):
        return self.conv(x)

# 创建网络和优化器
netD = DummyNet().to(device)
optimD = torch.optim.Adam(netD.parameters(), lr=0.001)

# 设置网络和优化器的梯度为零
netD.zero_grad()
optimD.zero_grad()

# 将标签扩展为与图像相同的空间尺寸 (batch_sz, num_classes, 32, 32)
image_one_hot_labelsF = image_one_hot_labelsF.repeat(1, 1, inps.shape[2], inps.shape[3])
image_one_hot_labelsR = image_one_hot_labelsR.repeat(1, 1, inps.shape[2], inps.shape[3])

# 输出扩展后的标签尺寸
print("\nAfter repeat:")
print("image_one_hot_labelsF shape:", image_one_hot_labelsF.shape)
print("image_one_hot_labelsR shape:", image_one_hot_labelsR.shape)
