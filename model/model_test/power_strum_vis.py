import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch



# 设置matplotlib使用支持中文的字体，或者使用英文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 尝试使用黑体，如果不可用则使用DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 创建一个示例图像用于演示
def create_sample_image(size=64):
    # 创建一个简单的测试图像 - 带有不同频率的正弦波图案
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    X, Y = np.meshgrid(x, y)

    # 组合不同频率的正弦波
    image = (np.sin(2 * X) + np.sin(5 * Y) + np.sin(0.5 * X + 0.5 * Y))

    # 归一化到0-255范围
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.astype(int)


# 可视化函数
def visualize_power_spectrum(images, E=100):
    """可视化功率谱分析的每个步骤"""
    # 确保输入是numpy数组
    if isinstance(images, torch.Tensor):
        image = images.detach().cpu().numpy().astype(int)
    else:
        image = images.astype(int)

    npix = image.shape[0]  # 假设是正方形图像

    # 设置图形布局
    fig = plt.figure(figsize=(15, 12))

    # 1. 原始图像
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.colorbar()

    # 2. 傅里叶变换与功率谱计算
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image) ** 2

    # 使用fftshift将零频率移到中心
    fourier_shifted = np.fft.fftshift(fourier_image)
    amplitude_shifted = np.fft.fftshift(fourier_amplitudes)

    # 显示傅里叶变换的实部
    plt.subplot(3, 3, 2)
    plt.imshow(np.log(np.abs(fourier_shifted) + 1), cmap='viridis')
    plt.title('傅里叶变换 (对数尺度)')
    plt.colorbar()

    # 显示功率谱
    plt.subplot(3, 3, 3)
    plt.imshow(np.log(amplitude_shifted + 1), cmap='hot')
    plt.title('功率谱 (对数尺度)')
    plt.colorbar()

    # 3. 频率网格计算
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    # 显示频率网格
    plt.subplot(3, 3, 4)
    plt.imshow(knrm, cmap='viridis')
    plt.title('频率模数网格')
    plt.colorbar()

    # 4. 扁平化处理
    knrm_flat = knrm.flatten()
    fourier_amplitudes_flat = fourier_amplitudes.flatten()

    # 显示扁平化后的频率与功率关系
    plt.subplot(3, 3, 5)
    plt.scatter(knrm_flat[:1000], np.log(fourier_amplitudes_flat[:1000] + 1),
                alpha=0.5, s=1)
    plt.xlabel('频率模数')
    plt.ylabel('对数功率')
    plt.title('频率-功率关系 (前1000个点)')

    # 5. 分箱统计
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    # 对功率谱按频率模数进行分箱统计
    Abins, bin_edges, binnumber = stats.binned_statistic(
        knrm_flat, fourier_amplitudes_flat, statistic="mean", bins=kbins)

    # 考虑二维空间的面积元素（环形区域面积）
    Abins_area = Abins * np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    # 绘制分箱统计结果
    plt.subplot(3, 3, 6)
    plt.plot(kvals, Abins, 'o-', label='平均功率')
    plt.plot(kvals, Abins_area, 's-', label='面积校正功率')
    plt.xlabel('频率')
    plt.ylabel('功率')
    plt.title('功率谱分箱统计')
    plt.legend()

    # 6. 获取前E个最重要的频率分量
    ind = np.argpartition(Abins_area, -E)[-E:]  # 获取功率最大的E个频率

    # 标记重要的频率分量
    plt.subplot(3, 3, 7)
    plt.plot(kvals, Abins_area, 'o-', label='面积校正功率')
    plt.plot(kvals[ind], Abins_area[ind], 'ro', label=f'前{E}个重要频率')
    plt.xlabel('频率')
    plt.ylabel('功率')
    plt.title(f'前{E}个重要频率分量')
    plt.legend()

    # 7. 在频率网格上标记重要频率
    important_freq_mask = np.zeros_like(knrm, dtype=bool)
    for i in ind:
        # 找到属于这个频率箱的所有点
        freq_min = kbins[i]
        freq_max = kbins[i + 1]
        mask = (knrm >= freq_min) & (knrm < freq_max)
        important_freq_mask |= mask

    plt.subplot(3, 3, 8)
    plt.imshow(important_freq_mask, cmap='gray')
    plt.title('重要频率区域')

    # 8. 在功率谱上叠加重要频率区域
    plt.subplot(3, 3, 9)
    plt.imshow(np.log(amplitude_shifted + 1), cmap='hot')
    plt.contour(important_freq_mask, levels=[0.5], colors='blue')
    plt.title('功率谱与重要频率区域')

    plt.tight_layout()
    plt.show()

    return ind


# 使用示例
if __name__ == "__main__":
    # 创建示例图像
    sample_image = create_sample_image(64)

    # 可视化功率谱分析过程
    important_freq_indices = visualize_power_spectrum(sample_image, E=20)

    print(f"前20个重要频率分量的索引: {important_freq_indices}")