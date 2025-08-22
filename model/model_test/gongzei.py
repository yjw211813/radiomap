import torch

# 检查可用GPU数量
print(f"可用GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 设置使用正确的设备
if torch.cuda.device_count() > 0:
    device = torch.device("cuda:2")  # 使用第一个可用GPU
else:
    device = torch.device("cpu")

print(f"使用设备: {device}")