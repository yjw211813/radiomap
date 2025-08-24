import torch
import numpy as np
import matplotlib.pyplot as plt

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath


def test_affine_prob_path():
    """
    测试 AffineProbPath 和 CondOTScheduler 的功能
    """
    print("=" * 50)
    print("测试 AffineProbPath 和 CondOTScheduler")
    print("=" * 50)

    # 创建路径实例
    path = AffineProbPath(scheduler=CondOTScheduler())

    # 创建测试数据
    batch_size = 5
    x_0 = torch.randn(batch_size, 3, 32, 32)  # 源数据（噪声）
    x_1 = torch.randn(batch_size, 3, 32, 32)  # 目标数据
    t = torch.rand(x_1.shape[0])  # 随机时间点

    print(f"源数据形状: {x_0.shape}")
    print(f"目标数据形状: {x_1.shape}")
    print(f"时间点: {t}")

    # 测试 sample 方法
    print("\n1. 测试 sample 方法:")
    path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

    print(f"路径样本形状: {path_sample.x_t.shape}")
    print(f"速度场形状: {path_sample.dx_t.shape}")

    # 验证样本计算是否正确
    # 根据 CondOTScheduler，x_t 应该等于 (1-t)*x_0 + t*x_1
    expected_x_t = (1 - t.view(-1, 1, 1, 1)) * x_0 + t.view(-1, 1, 1, 1) * x_1
    x_t_diff = torch.abs(path_sample.x_t - expected_x_t).mean()
    print(f"样本计算误差: {x_t_diff.item():.6f}")

    # 验证速度场计算是否正确
    # 根据 CondOTScheduler，dx_t 应该等于 x_1 - x_0
    expected_dx_t = x_1 - x_0
    dx_t_diff = torch.abs(path_sample.dx_t - expected_dx_t).mean()
    print(f"速度场计算误差: {dx_t_diff.item():.6f}")

    # 测试转换方法
    print("\n2. 测试转换方法:")

    # 测试 target_to_velocity
    velocity_from_target = path.target_to_velocity(x_1, path_sample.x_t, t)
    velocity_diff = torch.abs(velocity_from_target - path_sample.dx_t).mean()
    print(f"target_to_velocity 误差: {velocity_diff.item():.6f}")

    # 测试 epsilon_to_velocity
    # 首先计算 epsilon (噪声)
    epsilon = path.target_to_epsilon(x_1, path_sample.x_t, t)
    velocity_from_epsilon = path.epsilon_to_velocity(epsilon, path_sample.x_t, t)
    epsilon_velocity_diff = torch.abs(velocity_from_epsilon - path_sample.dx_t).mean()
    print(f"epsilon_to_velocity 误差: {epsilon_velocity_diff.item():.6f}")

    # 测试 velocity_to_target
    target_from_velocity = path.velocity_to_target(path_sample.dx_t, path_sample.x_t, t)
    target_diff = torch.abs(target_from_velocity - x_1).mean()
    print(f"velocity_to_target 误差: {target_diff.item():.6f}")

    # 测试 epsilon_to_target
    target_from_epsilon = path.epsilon_to_target(epsilon, path_sample.x_t, t)
    target_epsilon_diff = torch.abs(target_from_epsilon - x_1).mean()
    print(f"epsilon_to_target 误差: {target_epsilon_diff.item():.6f}")

    # 测试 velocity_to_epsilon
    epsilon_from_velocity = path.velocity_to_epsilon(path_sample.dx_t, path_sample.x_t, t)
    epsilon_velocity_diff = torch.abs(epsilon_from_velocity - epsilon).mean()
    print(f"velocity_to_epsilon 误差: {epsilon_velocity_diff.item():.6f}")

    # 测试 target_to_epsilon
    epsilon_from_target = path.target_to_epsilon(x_1, path_sample.x_t, t)
    epsilon_target_diff = torch.abs(epsilon_from_target - epsilon).mean()
    print(f"target_to_epsilon 误差: {epsilon_target_diff.item():.6f}")

    # 可视化测试
    print("\n3. 可视化测试:")
    visualize_path(path, x_0, x_1)

    # 边界条件测试
    print("\n4. 边界条件测试:")
    test_boundary_conditions(path, x_0, x_1)

    print("\n测试完成!")
    return True


def visualize_path(path, x_0, x_1):
    """
    可视化路径上的样本
    """
    # 选择第一个样本进行可视化
    x_0_sample = x_0[0:1]
    x_1_sample = x_1[0:1]

    # 生成多个时间点
    time_points = torch.linspace(0, 1, 10)

    # 存储路径上的样本
    path_samples = []

    for t in time_points:
        t_batch = torch.tensor([t])
        sample = path.sample(x_0=x_0_sample, x_1=x_1_sample, t=t_batch)
        path_samples.append(sample.x_t.detach().cpu().numpy())

    # 可视化
    plt.figure(figsize=(15, 5))

    # 显示源和目标
    plt.subplot(1, 3, 1)
    plt.imshow(x_0_sample[0].permute(1, 2, 0).cpu().numpy())
    plt.title("源数据 (X_0)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    # 显示中间点（选择第5个时间点）
    mid_point = path_samples[5][0].transpose(1, 2, 0)
    plt.imshow(mid_point)
    plt.title(f"中间点 (t=0.5)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(x_1_sample[0].permute(1, 2, 0).cpu().numpy())
    plt.title("目标数据 (X_1)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("path_visualization.png")
    plt.close()

    print("可视化已保存到 path_visualization.png")


def test_boundary_conditions(path, x_0, x_1):
    """
    测试边界条件
    """
    # 测试 t=0
    t0 = torch.zeros(x_0.shape[0])
    sample_t0 = path.sample(x_0=x_0, x_1=x_1, t=t0)
    t0_diff = torch.abs(sample_t0.x_t - x_0).mean()
    print(f"t=0 时，X_t 应该等于 X_0，误差: {t0_diff.item():.6f}")

    # 测试 t=1
    t1 = torch.ones(x_0.shape[0])
    sample_t1 = path.sample(x_0=x_0, x_1=x_1, t=t1)
    t1_diff = torch.abs(sample_t1.x_t - x_1).mean()
    print(f"t=1 时，X_t 应该等于 X_1，误差: {t1_diff.item():.6f}")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行测试
    test_affine_prob_path()