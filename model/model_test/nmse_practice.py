import torch


def nmse(outputs, targets):
    """
    计算NMSE（归一化均方误差）
    参数:
        outputs: 网络输出，形状(batch_size, 1, H, W)
        targets: 目标值，形状(batch_size, 1, H, W)
    返回:
        NMSE值
    """
    # 计算分子：预测值与真实值的均方误差
    numerator = torch.sum((outputs - targets) ** 2)

    # 计算分母：真实值的平方和
    denominator = torch.sum(targets ** 2)

    # 避免除以零
    if denominator == 0:
        return torch.tensor(float('nan'))

    return numerator / denominator


# 使用示例
outputs = torch.randn(8, 1, 256, 256)  # 模拟网络输出
targets = torch.randn(8, 1, 256, 256)  # 模拟目标值

nmse_value = nmse(outputs, targets)
print(f"NMSE: {nmse_value.item()}")