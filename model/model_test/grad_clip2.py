import torch
import torch.nn as nn


# 创建一个简单的模型类来包装我们的张量
class SimpleModel(nn.Module):
    def __init__(self, x, y):
        super(SimpleModel, self).__init__()
        # 将张量注册为模型参数
        self.x = nn.Parameter(x)
        self.y = nn.Parameter(y)

    def forward(self):
        # 模拟网络计算过程
        z = self.x ** 2 + self.y ** 3
        return z.sum()


# 构造两个Tensor
x = torch.tensor([99.0, 108.0], requires_grad=True)
y = torch.tensor([45.0, 75.0], requires_grad=True)

# 创建模型
model = SimpleModel(x, y)

# 模拟网络计算过程
z = model()

# 反向传播
z.backward()

# 得到原始梯度
print("原始梯度:")
print(f"gradient of x is: {model.x.grad}")
print(f"gradient of y is: {model.y.grad}")

# 计算原始梯度范数
x_L2norm = torch.sum(model.x.grad ** 2) ** 0.5
y_L2norm = torch.sum(model.y.grad ** 2) ** 0.5
total_norm = (x_L2norm ** 2 + y_L2norm ** 2) ** 0.5
print(f"原始梯度总范数: {total_norm.item()}")

# 应用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

print("\n应用梯度裁剪后的梯度:")
print(f"gradient of x is: {model.x.grad}")
print(f"gradient of y is: {model.y.grad}")

# 计算裁剪后的梯度范数
x_L2norm_clipped = torch.sum(model.x.grad ** 2) ** 0.5
y_L2norm_clipped = torch.sum(model.y.grad ** 2) ** 0.5
total_norm_clipped = (x_L2norm_clipped ** 2 + y_L2norm_clipped ** 2) ** 0.5
print(f"裁剪后梯度总范数: {total_norm_clipped.item()}")

# 验证裁剪是否正确
print(f"\n验证裁剪是否正确:")
print(f"目标范数: 2.0")
print(f"实际范数: {total_norm_clipped.item()}")
print(f"误差: {abs(total_norm_clipped.item() - 2.0)}")

# 计算手动裁剪系数
max_norm = 2
clip_coef = max_norm / total_norm

print(f"\n手动计算裁剪系数: {clip_coef.item()}")
print(f"手动裁剪后的x梯度: {model.x.grad / clip_coef}")
print(f"手动裁剪后的y梯度: {model.y.grad / clip_coef}")