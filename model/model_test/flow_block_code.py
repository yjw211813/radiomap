import torch
import matplotlib.pyplot as plt

def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size, ), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)
    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45
    return data.float()

# 生成数据
data = inf_train_gen(batch_size=2000, device="cpu")
x = data[:, 0].numpy()
y = data[:, 1].numpy()

# 可视化
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=10, alpha=0.6, c='blue', edgecolors='w')
plt.title("Generated Data Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(alpha=0.3)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()