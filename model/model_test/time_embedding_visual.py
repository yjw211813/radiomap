import torch
import math
import matplotlib.pyplot as plt

d_model = 512  # 假设 d_model = 512

# 生成线性间隔值（取负指数前）
emb_linear = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
# 取负指数后
emb_exp = torch.exp(-emb_linear)

# 绘制图形
plt.figure(figsize=(12, 5))

# 1. 线性间隔值（取负指数前）
plt.subplot(1, 2, 1)
plt.plot(emb_linear.numpy(), 'o-', color='blue')
plt.title("Linear  Spacing (before exp(-x))")
plt.xlabel("Position  (i)")
plt.ylabel("Value")
plt.grid(True)

# 2. 负指数变换后（0.0001 ~ 1）
plt.subplot(1, 2, 2)
plt.plot(emb_exp.numpy(), 'o-', color='red')
plt.title("After  exp(-x)")
plt.xlabel("Position  (i)")
plt.ylabel("Value")
# 删除设置对数坐标系的代码
plt.grid(True)

plt.tight_layout()
plt.show()
