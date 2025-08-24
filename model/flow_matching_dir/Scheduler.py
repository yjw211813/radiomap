from torch.optim.lr_scheduler import _LRScheduler

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
