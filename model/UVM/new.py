import torch
from mamba_ssm import Mamba

# 检查CUDA可用性
assert torch.cuda.is_available(), "CUDA不可用"
device = torch.device("cuda:2")

# 初始化Mamba模型并转移到GPU
model = Mamba(
    d_model=256,  # 示例维度
    d_state=16,
    d_conv=4,
    expand=2
).to(device)

# 验证模型参数是否在GPU上
assert next(model.parameters()).is_cuda, "模型参数未转移到GPU"