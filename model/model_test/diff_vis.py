import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 创建一个简单的4x4图像张量作为示例 (batch_size=1, channels=1, height=4, width=4)
y_hat = torch.randn(1, 1, 4, 4)
print("原始图像张量:")
print(y_hat.squeeze().numpy())

# 计算水平方向的差异 (i方向)
diff_i = torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1])
print("\n水平方向差异 (i方向):")
print(diff_i.squeeze().numpy())

# 计算垂直方向的差异 (j方向)
diff_j = torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :])
print("\n垂直方向差异 (j方向):")
print(diff_j.squeeze().numpy())

# 计算总变差损失
TV_WEIGHT = 0.001
tv_loss = TV_WEIGHT * (torch.sum(diff_i) + torch.sum(diff_j))
print(f"\n总变差损失: {tv_loss.item():.6f}")
# 创建可视化图表

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('总变差(TV)损失计算可视化', fontsize=16)

# 原始图像
orig_img = y_hat.squeeze().numpy()
im0 = axes[0, 0].imshow(orig_img, cmap='viridis')
axes[0, 0].set_title('原始图像')
axes[0, 0].set_xticks(np.arange(4))
axes[0, 0].set_yticks(np.arange(4))
for i in range(4):
    for j in range(4):
        axes[0, 0].text(j, i, f'{orig_img[i, j]:.2f}', ha='center', va='center', color='white')
plt.colorbar(im0, ax=axes[0, 0])

# 水平方向差异计算
diff_i_np = diff_i.squeeze().numpy()
im1 = axes[0, 1].imshow(diff_i_np, cmap='Reds')
axes[0, 1].set_title('水平方向差异 (i方向)')
axes[0, 1].set_xticks(np.arange(3))
axes[0, 1].set_yticks(np.arange(4))
for i in range(4):
    for j in range(3):
        axes[0, 1].text(j, i, f'{diff_i_np[i, j]:.2f}', ha='center', va='center', color='white')
plt.colorbar(im1, ax=axes[0, 1])

# 垂直方向差异计算
diff_j_np = diff_j.squeeze().numpy()
im2 = axes[0, 2].imshow(diff_j_np, cmap='Blues')
axes[0, 2].set_title('垂直方向差异 (j方向)')
axes[0, 2].set_xticks(np.arange(4))
axes[0, 2].set_yticks(np.arange(3))
for i in range(3):
    for j in range(4):
        axes[0, 2].text(j, i, f'{diff_j_np[i, j]:.2f}', ha='center', va='center', color='white')
plt.colorbar(im2, ax=axes[0, 2])

# 绘制计算示意图 - 水平差异
axes[1, 0].imshow(orig_img, cmap='viridis', alpha=0.3)
axes[1, 0].set_title('水平差异计算示意图')
axes[1, 0].set_xticks(np.arange(4))
axes[1, 0].set_yticks(np.arange(4))

# 添加水平差异的箭头和矩形
for i in range(4):
    for j in range(3):
        axes[1, 0].add_patch(Rectangle((j, i), 2, 1, fill=False, edgecolor='red', linewidth=2))
        axes[1, 0].arrow(j+0.3, i+0.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
        axes[1, 0].arrow(j+1.7, i+0.5, -0.4, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')

# 绘制计算示意图 - 垂直差异
axes[1, 1].imshow(orig_img, cmap='viridis', alpha=0.3)
axes[1, 1].set_title('垂直差异计算示意图')
axes[1, 1].set_xticks(np.arange(4))
axes[1, 1].set_yticks(np.arange(4))

# 添加垂直差异的箭头和矩形
for i in range(3):
    for j in range(4):
        axes[1, 1].add_patch(Rectangle((j, i), 1, 2, fill=False, edgecolor='blue', linewidth=2))
        axes[1, 1].arrow(j+0.5, i+0.3, 0, 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        axes[1, 1].arrow(j+0.5, i+1.7, 0, -0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

# 显示最终损失值
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.7, f'水平差异总和: {torch.sum(diff_i).item():.4f}',
                ha='center', va='center', fontsize=14, color='red')
axes[1, 2].text(0.5, 0.5, f'垂直差异总和: {torch.sum(diff_j).item():.4f}',
                ha='center', va='center', fontsize=14, color='blue')
axes[1, 2].text(0.5, 0.3, f'TV权重: {TV_WEIGHT}',
                ha='center', va='center', fontsize=14)
axes[1, 2].text(0.5, 0.1, f'总变差损失: {tv_loss.item():.6f}',
                ha='center', va='center', fontsize=16, weight='bold')

plt.tight_layout()
plt.show()