import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


# 定义热身调度器
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier          # 学习率增加的倍数
        self.warm_epoch = warm_epoch         # 预热阶段的epoch数
        self.after_scheduler = after_scheduler# 预热后使用的调度器
        self.finished = False                 # 标记预热是否完成
        self.last_epoch = None                # 当前epoch数
        self.base_lrs = None                  # 初始学习率列表
        super().__init__(optimizer)           # 调用父类初始化

    def get_lr(self):
        # 如果已经过了预热阶段
        if self.last_epoch > self.warm_epoch:
            if self.after_scheduler:        # 如果有后续调度器
                if not self.finished:       # 如果还没设置后续调度器的基准学习率
                    # 设置后续调度器的基准学习率为预热结束时的学习率
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warm_epoch + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warm_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


# 创建一个简单的模型和优化器
model = nn.Linear(10, 1)
optimizer = SGD(model.parameters(), lr=0.01)  # 初始学习率为0.01

# 设置总epoch数
total_epoch = 100

# 创建学习率调度器
cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=total_epoch - total_epoch // 10,
    eta_min=0,
    last_epoch=-1
)

warmUpScheduler = GradualWarmupScheduler(
    optimizer=optimizer,
    multiplier=2.5,  # 热身阶段最终会将学习率提高到初始值的2.5倍
    warm_epoch=total_epoch // 10,  # 热身阶段占10%的总epoch
    after_scheduler=cosineScheduler
)

# 记录每个epoch的学习率
learning_rates = []

# 模拟训练过程
for epoch in range(total_epoch):
    # 计算梯度
    optimizer.zero_grad()


    # 更新参数
    optimizer.step()

    # 更新学习率
    warmUpScheduler.step()

    # 获取更新后的学习率（用于下一个epoch）
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

# 绘制学习率变化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(total_epoch), learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule: Warmup + Cosine Annealing')
plt.grid(True)

# 标记热身阶段结束的位置
warmup_end = total_epoch // 10
plt.axvline(x=warmup_end, color='r', linestyle='--', alpha=0.5)
plt.text(warmup_end + 1, max(learning_rates) * 0.8, f'Warmup End (Epoch {warmup_end})', color='r')

plt.show()

# 打印一些关键信息
print(f"初始学习率: {learning_rates[0]:.6f}")
print(f"热身阶段结束时的学习率: {learning_rates[warmup_end]:.6f}")
print(f"最终学习率: {learning_rates[-1]:.6f}")
print(f"学习率变化倍数: {learning_rates[warmup_end] / learning_rates[0]:.2f}x")