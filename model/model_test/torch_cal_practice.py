import torch

# 创建可验证的张量（非随机）
# tensor_b: 每个通道有唯一标识值
tensor_b = torch.zeros(32, 8, 32, 32)
for ch in range(8):
    tensor_b[:, ch, :, :] = ch  # 每个通道填充其通道索引值

# 原始张量（内容不重要）
tensor_a = torch.randn(32, 32, 32, 32)

# 沿第1维度复制4次
tensor_b_expanded = tensor_b.repeat(1, 4, 1, 1)  # 形状 (32,32,32,32)

# ===== 验证1: 检查形状 =====
print("原始tensor_b形状:", tensor_b.shape)
print("扩展后形状:", tensor_b_expanded.shape)

# ===== 验证2: 检查通道值 =====
# 检查第一个样本的所有通道
print("\n通道值验证:")
for i in range(0, 32, 8):  # 每8个通道为一组
    # 获取当前组的通道值
    group_values = tensor_b_expanded[0, i:i + 8, 0, 0]

    # 预期值: 0,1,2,3,4,5,6,7
    expected = torch.arange(0, 8, dtype=torch.float)

    print(f"通道 {i:2d} 到 {i + 7:2d}: 值 = {group_values.tolist()}")
    print(f"是否符合原始通道序列: {torch.equal(group_values, expected)}")
    print()

# ===== 验证3: 检查所有样本和位置 =====
print("\n全面验证:")
all_correct = True

# 检查每个样本
for sample in range(32):
    # 检查每个空间位置
    for h in range(32):
        for w in range(32):
            # 检查每8个通道的组
            for group_start in range(0, 32, 8):
                # 获取当前组的通道值
                actual = tensor_b_expanded[sample, group_start:group_start + 8, h, w]

                # 预期值总是0-7
                expected = torch.arange(0, 8, dtype=torch.float)

                if not torch.equal(actual, expected):
                    print(f"错误在: 样本={sample}, 位置=({h},{w}), 通道组={group_start}-{group_start + 7}")
                    all_correct = False
                    break
            if not all_correct:
                break
        if not all_correct:
            break
    if not all_correct:
        break

if all_correct:
    print("所有位置验证通过：每个8通道组都正确复制了原始通道序列")
else:
    print("验证未通过，存在不一致数据")

# ===== 验证4: 可视化检查 =====
import matplotlib.pyplot as plt

# 创建可视化数据
sample_idx = 0
position = (15, 15)  # 选择中间位置

# 提取扩展张量的通道值
channel_values = tensor_b_expanded[sample_idx, :, position[0], position[1]]

# 绘制通道值变化
plt.figure(figsize=(12, 6))
plt.plot(channel_values.numpy())
plt.title("通道值变化 (每8个通道一组重复)")
plt.xlabel("通道索引")
plt.ylabel("通道值")
plt.axvline(x=7.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=15.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=23.5, color='r', linestyle='--', alpha=0.5)
plt.grid(True)
plt.show()