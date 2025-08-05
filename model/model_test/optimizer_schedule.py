import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
import warnings

class DynamicLRScheduler(_LRScheduler):
    """
    动态学习率调度器，在 [lr_min, lr_max] 范围内变化

    参数:
        optimizer (Optimizer): 优化器对象
        lr_min (float): 最小学习率
        lr_max (float): 最大学习率
        warmup_epochs (int): 学习率上升阶段 epoch 数
        decay_epochs (int): 学习率下降阶段 epoch 数
        last_epoch (int): 上一个 epoch 索引（默认为 -1）
        verbose (bool): 是否打印更新信息
    """

    def __init__(self, optimizer, lr_min, lr_max, warmup_epochs=5,
                 decay_epochs=20, last_epoch=-1, verbose=False):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.total_epochs = warmup_epochs + decay_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """计算当前 epoch 的学习率"""
        if not self._get_lr_called_within_step:
            warnings.warn("需要在 scheduler.step() 之前调用", UserWarning)

        epoch = self.last_epoch

        # 1. 预热阶段：线性上升
        if epoch < self.warmup_epochs:
            progress = epoch / max(1, self.warmup_epochs)
            lr = self.lr_min + (self.lr_max - self.lr_min) * progress

        # 2. 衰减阶段：余弦退火
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.decay_epochs)
            # 使用余弦函数平滑衰减
            cosine_decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
            lr = self.lr_min + (self.lr_max - self.lr_min) * cosine_decay

        return [lr for _ in self.optimizer.param_groups]


# 使用示例
if __name__ == "__main__":
    # 1. 创建模型和优化器
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,  # 初始学习率（将被调度器覆盖）
        betas=(0.9, 0.999),
        weight_decay=0,
        amsgrad=False
    )

    # 2. 初始化调度器
    scheduler = DynamicLRScheduler(
        optimizer,
        lr_min=1e-6,  # 最小学习率
        lr_max=1e-3,  # 最大学习率
        warmup_epochs=5,  # 前5个epoch学习率上升
        decay_epochs=20  # 后20个epoch学习率下降
    )

    # 3. 训练循环
    num_epochs = 25
    for epoch in range(num_epochs):
        # 训练步骤...
        # loss.backward()
        # optimizer.step()

        # 更新学习率
        scheduler.step()

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs} \t Learning Rate: {current_lr:.2e}")