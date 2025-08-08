import math
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter


class WarmupMultiStepExponentialLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, mid_epoch, total_epochs,
                 lr_init=1e-6, lr_max=1e-3, lr_mid=1e-4, lr_final=1e-6,
                 last_epoch=-1):
        """
        三段式学习率调整器：
        1. 线性预热 (0 ≤ epoch < warmup_epochs)
        2. 快速指数衰减 (warmup_epochs ≤ epoch < mid_epoch)
        3. 缓慢指数衰减 (mid_epoch ≤ epoch)

        参数:
        - warmup_epochs: 预热阶段轮数 (30)
        - mid_epoch: 第一阶段衰减结束轮数 (100)
        - total_epochs: 总训练轮数 (≥mid_epoch)
        - lr_init: 初始学习率 (1e-6)
        - lr_max: 预热结束学习率 (1e-3)
        - lr_mid: 中间阶段学习率 (1e-4)
        - lr_final: 最终学习率 (1e-6)
        """
        self.warmup_epochs = warmup_epochs
        self.mid_epoch = mid_epoch
        self.total_epochs = total_epochs
        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_mid = lr_mid
        self.lr_final = lr_final

        # 计算衰减率
        self.first_decay_epochs = mid_epoch - warmup_epochs
        self.decay_rate1 = (lr_mid / lr_max) ** (1 / self.first_decay_epochs)

        self.second_decay_epochs = total_epochs - mid_epoch
        if self.second_decay_epochs > 0:
            self.decay_rate2 = (lr_final / lr_mid) ** (1 / self.second_decay_epochs)
        else:
            self.decay_rate2 = 1.0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        # 1. 线性预热阶段 (0 ≤ epoch < warmup_epochs)
        if epoch < self.warmup_epochs:
            progress = epoch / max(1, self.warmup_epochs)
            return [self.lr_init + (self.lr_max - self.lr_init) * progress
                    for _ in self.optimizer.param_groups]

        # 2. 快速指数衰减阶段 (warmup_epochs ≤ epoch < mid_epoch)
        elif epoch < self.mid_epoch:
            steps = epoch - self.warmup_epochs
            return [self.lr_max * (self.decay_rate1 ** steps)
                    for _ in self.optimizer.param_groups]

        # 3. 缓慢指数衰减阶段 (mid_epoch ≤ epoch)
        else:
            steps = epoch - self.mid_epoch
            return [self.lr_mid * (self.decay_rate2 ** steps)
                    for _ in self.optimizer.param_groups]

    # 增强版可视化工具
    def visualize_schedule(self, save_path=None, dpi=100):
        """
        增强版学习率可视化，包含线性刻度和对数刻度视图

        参数:
        save_path (str): 图片保存路径（可选）
        dpi (int): 图片分辨率

        返回:
        lrs (list): 各epoch对应的学习率列表
        """
        # 计算所有epoch的学习率
        epochs = list(range(self.total_epochs))
        lrs = []

        # 保存原始状态
        original_last_epoch = self.last_epoch

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

        # 定义关键点
        key_points = [
            (0, "Start"),
            (self.warmup_epochs - 1, f"Warmup End\n({lrs[self.warmup_epochs - 1]:.2e})"),
            (self.mid_epoch - 1, f"Mid Decay End\n({lrs[self.mid_epoch - 1]:.2e})"),
            (self.total_epochs - 1, f"Final LR\n({lrs[self.total_epochs - 1]:.2e})")
        ]

        # 添加中点（如果存在）
        mid_point = (self.warmup_epochs + self.mid_epoch) // 2
        if mid_point < self.mid_epoch:
            key_points.insert(2, (mid_point, f"Fast Decay Mid\n({lrs[mid_point]:.2e})"))

        # 添加第二阶段中点
        mid_point2 = (self.mid_epoch + self.total_epochs) // 2
        if mid_point2 < self.total_epochs:
            key_points.insert(4, (mid_point2, f"Slow Decay Mid\n({lrs[mid_point2]:.2e})"))

        # 标注关键点
        for epoch, label in key_points:
            if epoch < len(lrs):  # 确保不越界
                ax1.scatter(epoch, lrs[epoch], c='red', s=100, zorder=5)
                ax1.annotate(label, (epoch, lrs[epoch]),
                             xytext=(10, 20), textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', color='red'),
                             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

        # 添加区域标注
        ax1.axvspan(0, self.warmup_epochs, alpha=0.1, color='green', label='Warmup Phase')
        ax1.axvspan(self.warmup_epochs, self.mid_epoch, alpha=0.1, color='orange', label='Fast Decay Phase')
        ax1.axvspan(self.mid_epoch, self.total_epochs, alpha=0.1, color='purple', label='Slow Decay Phase')
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
            if epoch < len(lrs):  # 确保不越界
                ax2.scatter(epoch, lrs[epoch], c='blue', s=100, zorder=5)
                ax2.annotate(f"{lrs[epoch]:.2e}", (epoch, lrs[epoch]),
                             xytext=(10, 10), textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', color='blue'),
                             bbox=dict(boxstyle='round,pad=0.3', fc='cyan', alpha=0.3))

        # 添加学习率范围标注
        for lr_val, label in [(self.lr_init, "Init LR"),
                              (self.lr_max, "Max LR"),
                              (self.lr_mid, "Mid LR"),
                              (self.lr_final, "Final LR")]:
            if lr_val > 0:  # 确保正值
                ax2.axhline(y=lr_val, color='gray', linestyle='--', alpha=0.5)
                ax2.text(self.total_epochs * 0.8, lr_val * (1.2 if lr_val < 1e-4 else 0.8),
                         f"{label}: {lr_val:.1e}", fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.8))

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
    # 创建虚拟优化器
    import torch

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 初始化调度器
    scheduler = WarmupMultiStepExponentialLR(
        optimizer=optimizer,
        warmup_epochs=30,
        mid_epoch=100,
        total_epochs=300,
        lr_init=1e-6,
        lr_max=1e-3,
        lr_mid=1e-4,
        lr_final=1e-6
    )

    # 可视化学习率曲线
    lrs = scheduler.visualize_schedule(save_path=None)

    # 打印关键点值
    print(f"0轮学习率: {lrs[0]:.2e} (初始学习率)")
    print(f"30轮学习率: {lrs[29]:.2e} (预热结束，目标: 1e-3)")
    print(f"65轮学习率: {lrs[64]:.2e} (快速衰减中点)")
    print(f"100轮学习率: {lrs[99]:.2e} (快速衰减结束，目标: 1e-4)")
    print(f"200轮学习率: {lrs[199]:.2e} (缓慢衰减中点)")
    print(f"300轮学习率: {lrs[299]:.2e} (训练结束，目标: 1e-6)")