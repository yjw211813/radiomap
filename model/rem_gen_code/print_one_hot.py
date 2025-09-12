import torch


# 假设 ohe_vector_from_labels 是一个将标签转化为 one-hot 编码的函数
def ohe_vector_from_labels(labels, num_classes):
    """
    将标签转化为 one-hot 编码格式

    参数:
        labels (Tensor): 输入标签，类型为 (batch_sz,)
        num_classes (int): 类别数量

    返回:
        Tensor: one-hot 编码后的张量，形状为 (batch_sz, num_classes)
    """
    # 创建一个形状为 (batch_sz, num_classes) 的全零张量
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    # 根据标签的位置设置为1
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot


# 假设当前的设备是 CPU 或 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch_sz 定义了批次的大小
batch_sz = 4  # 可以根据需要更改批次大小

# 示例：标签为 0 的 one-hot 编码
one_hot_labelsF = ohe_vector_from_labels(torch.tensor([0] * batch_sz).to(device), 2)

# 示例：标签为 1 的 one-hot 编码
one_hot_labelsR = ohe_vector_from_labels(torch.tensor([1] * batch_sz).to(device), 2)

# 打印结果
print("One-hot labels F:\n", one_hot_labelsF)
print("One-hot labels R:\n", one_hot_labelsR)
