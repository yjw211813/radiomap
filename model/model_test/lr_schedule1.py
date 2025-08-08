import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter
from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings
# 创建虚拟优化器
import torch
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
                 decay_epochs=20, last_epoch=-1):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.total_epochs = warmup_epochs + decay_epochs
        super().__init__(optimizer, last_epoch)

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

    def visualize_schedule(self, total_epochs=None, save_path=None, dpi=100):
        """
        可视化学习率变化曲线

        参数:
        total_epochs (int): 要可视化的总epoch数（默认使用调度器设置的总epoch）
        save_path (str): 图片保存路径（可选）
        dpi (int): 图片分辨率

        返回:
        lrs (list): 各epoch对应的学习率列表
        """
        # 确定要显示的总epoch数
        if total_epochs is None:
            total_epochs = self.total_epochs
        else:
            total_epochs = min(total_epochs, self.total_epochs)

        # 保存原始状态
        original_last_epoch = self.last_epoch

        # 计算所有epoch的学习率
        epochs = list(range(total_epochs))
        lrs = []

        for epoch in epochs:
            self.last_epoch = epoch
            lr = self.get_lr()[0]
            lrs.append(lr)

        # 恢复原始状态
        self.last_epoch = original_last_epoch

        # 创建图形
        plt.figure(figsize=(12, 8), dpi=dpi)

        # 主坐标轴（线性刻度）
        ax1 = plt.subplot(211)
        ax1.plot(epochs, lrs, 'b-', linewidth=2.5)
        ax1.set_title('Learning Rate Schedule (Linear Scale)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 标注关键点
        warmup_end = min(self.warmup_epochs, total_epochs - 1)
        decay_mid = min(self.warmup_epochs + self.decay_epochs // 2, total_epochs - 1)
        end_point = min(self.total_epochs - 1, total_epochs - 1)

        key_points = [
            (0, "Start"),
            (warmup_end, f"Warmup End\n({lrs[warmup_end]:.2e})"),
            (decay_mid, f"Mid Decay\n({lrs[decay_mid]:.2e})"),
            (end_point, f"Final LR\n({lrs[end_point]:.2e})")
        ]

        for epoch, label in key_points:
            ax1.scatter(epoch, lrs[epoch], c='red', s=100, zorder=5)
            ax1.annotate(label, (epoch, lrs[epoch]),
                         xytext=(10, 20), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='red'),
                         bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

        # 添加区域标注
        ax1.axvspan(0, self.warmup_epochs, alpha=0.1, color='green', label='Warmup Phase')
        ax1.axvspan(self.warmup_epochs, total_epochs, alpha=0.1, color='orange', label='Decay Phase')
        ax1.legend(loc='upper right')

        # 副坐标轴（对数刻度）
        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(epochs, lrs, 'r-', linewidth=2.5)
        ax2.set_title('Learning Rate Schedule (Log Scale)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.7, which='both')

        # 设置对数刻度的格式
        ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax2.yaxis.set_minor_formatter(NullFormatter())

        # 标注关键点（对数视图）
        for epoch, label in key_points:
            ax2.scatter(epoch, lrs[epoch], c='blue', s=100, zorder=5)
            ax2.annotate(f"{lrs[epoch]:.2e}", (epoch, lrs[epoch]),
                         xytext=(10, 10), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='blue'),
                         bbox=dict(boxstyle='round,pad=0.3', fc='cyan', alpha=0.3))

        # 添加学习率范围标注
        ax2.axhline(y=self.lr_min, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.lr_max, color='gray', linestyle='--', alpha=0.5)
        ax2.text(total_epochs * 0.8, self.lr_min * 1.2, f"Min LR: {self.lr_min:.1e}",
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        ax2.text(total_epochs * 0.8, self.lr_max * 0.8, f"Max LR: {self.lr_max:.1e}",
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()

        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()

        return lrs


# 使用示例
if __name__ == "__main__":

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 初始化调度器（使用您提供的参数）
    scheduler = DynamicLRScheduler(
        optimizer=optimizer,
        lr_min=1e-6,
        lr_max=1e-3,
        warmup_epochs=30,
        decay_epochs=70  # 总epochs=100
    )

    # 可视化学习率曲线
    lrs = scheduler.visualize_schedule(total_epochs=120, save_path=None)

    # 打印关键点值
    print(f"0轮学习率: {lrs[0]:.2e} (初始学习率)")
    print(f"30轮学习率: {lrs[30]:.2e} (预热结束，目标: 1e-3)")
    print(f"60轮学习率: {lrs[60]:.2e} (衰减中点，目标: 3e-4)")
    print(f"100轮学习率: {lrs[99]:.2e} (训练结束，目标: 1e-4)")