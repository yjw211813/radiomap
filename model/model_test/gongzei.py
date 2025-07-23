import torch

A = torch.randn(8, 64, 64, 64)  # 形状 (8, 64, 64, 64)
B = torch.randn(8, 1, 64, 64)   # 形状 (8, 1, 64, 64)

result = A * B  # 自动广播，结果形状 (8, 64, 64, 64)
print(result.shape)  # 输出: torch.Size([8, 64, 64, 64])