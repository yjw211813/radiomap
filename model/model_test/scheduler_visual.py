import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim import lr_scheduler


def visualize_lr_scheduler(total_epochs=100, step_size=30, gamma=0.1, lr=1e-4):
    """
    可视化学习率调度器的学习率变化

    参数:
    total_epochs: 总训练轮数
    step_size: 学习率调整步长
    gamma: 学习率调整系数
    lr: 初始学习率
    """
    # 创建一个虚拟模型参数
    dummy_param = torch.nn.Parameter(torch.randn(10, 10))

    # 创建优化器
    optimizer = optim.Adam([dummy_param], lr=lr)

    # 创建学习率调度器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 记录每个epoch的学习率
    learning_rates = []

    # 模拟训练过程
    for epoch in range(total_epochs):
        # 记录当前学习率
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 更新学习率（通常在每个epoch的优化步骤之后调用）
        scheduler.step()

    # 绘制学习率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(total_epochs), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'StepLR Scheduler (step_size={step_size}, gamma={gamma})')
    plt.grid(True)
    plt.yscale('log')  # 使用对数坐标更清晰地显示变化
    plt.show()

    return learning_rates


# 使用示例
if __name__ == "__main__":
    # 可视化默认参数的学习率变化
    lr_history = visualize_lr_scheduler(total_epochs=100, step_size=30, gamma=0.1, lr=1e-4)

    # 也可以尝试不同的参数
    # lr_history = visualize_lr_scheduler(total_epochs=100, step_size=20, gamma=0.5, lr=1e-3)