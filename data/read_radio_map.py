import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

def read_radio_map_data(file_path):
    """读取无线电地图数据"""
    samp_datas = []
    origi_datas = []

    for file_name in os.listdir(file_path):
        if file_name.endswith('.h5'):
            with h5py.File(os.path.join(file_path, file_name), 'r') as f:
                samp_datas.append(f['samp'][:])
                origi_datas.append(f['origi'][:])

    return np.array(samp_datas), np.array(origi_datas)


def visualize_comparison(origi, samp, sample_idx=0):
    """可视化对比原始数据和采样数据"""
    plt.figure(figsize=(15, 10))

    # 原始数据
    plt.subplot(2, 3, 1)
    plt.imshow(origi[sample_idx], cmap='viridis')
    plt.title(f'Original Data\nSample {sample_idx}')
    plt.colorbar()

    # 采样数据（含缺失值）
    plt.subplot(2, 3, 2)
    plt.imshow(samp[sample_idx], cmap='viridis')
    plt.title(f'Sampled Data\nSample {sample_idx}')
    plt.colorbar()

    # 缺失值分布
    plt.subplot(2, 3, 3)
    missing_mask = (samp[sample_idx] == -2000)
    plt.imshow(missing_mask, cmap='gray')
    plt.title(f'Missing Values\n({np.sum(missing_mask)} pixels)')

    # 数据差异热力图
    plt.subplot(2, 3, 4)
    diff = np.abs(origi[sample_idx] - samp[sample_idx])
    # 将缺失值差异设为0以便可视化
    diff[missing_mask] = 0
    plt.imshow(diff, cmap='hot')
    plt.title('Absolute Difference (Non-missing)')
    plt.colorbar()

    # 随机选取5个位置验证
    plt.subplot(2, 3, 5)
    non_missing_indices = np.argwhere(~missing_mask)
    selected_indices = non_missing_indices[np.random.choice(len(non_missing_indices), 5, replace=False)]

    for i, (y, x) in enumerate(selected_indices):
        orig_val = origi[sample_idx, y, x]
        samp_val = samp[sample_idx, y, x]
        match = "MATCH" if orig_val == samp_val else "MISMATCH"
        plt.text(0.1, 0.8 - i * 0.15,
                 f"Position ({y},{x}): Orig={orig_val:.2f}, Samp={samp_val:.2f} → {match}",
                 fontsize=10, transform=plt.gca().transAxes)

    plt.axis('off')
    plt.title('Random Position Verification')

    plt.tight_layout()
    plt.show()


def read_radio_map_data(file_path):
    """
    读取由 generate_radio_map_data 生成的 HDF5 文件
    Args:
        file_path (str): HDF5 文件路径
    Returns:
        tuple: (samp_datas, origi_datas) 两个 NumPy 数组
    """
    with h5py.File(file_path, 'r') as f:
        samp_datas = np.array(f['samp_datas'])  # 读取采样数据
        origi_datas = np.array(f['origi_datas'])  # 读取原始数据
    return samp_datas, origi_datas


# 示例用法
if __name__ == "__main__":

    file_path = "./radio_map_train"

    # 读取数据
    samp_datas, origi_datas = read_radio_map_data(file_path)

    # 打印基本信息
    print(f"samp_datas shape: {samp_datas.shape}, dtype: {samp_datas.dtype}")
    print(f"origi_datas shape: {origi_datas.shape}, dtype: {origi_datas.dtype}")

    # 验证非缺失位置数据一致性
    print("\n验证非缺失位置数据一致性:")
    for i in range(5):
        mask = (samp_datas[i] != -2000)
        matching = (samp_datas[i][mask] == origi_datas[i][mask])

        match_percent = np.mean(matching) * 100
        num_non_missing = np.sum(mask)

        print(f"样本 {i}: {num_non_missing}个非缺失点中，{match_percent:.2f}% 匹配")

        if not np.all(matching):
            mismatch_indices = np.where(~matching)
            print(f"  发现 {len(mismatch_indices[0])} 个不匹配点，示例:")
            for j in range(min(3, len(mismatch_indices[0]))):
                y, x = mismatch_indices[0][j], mismatch_indices[1][j]
                print(f"    位置 ({y}, {x}): "
                      f"原始值={origi_datas[i, y, x]:.4f}, "
                      f"采样值={samp_datas[i, y, x]:.4f}")

    # 可视化前3个样本
    for i in range(3):
        visualize_comparison(origi_datas, samp_datas, sample_idx=i)