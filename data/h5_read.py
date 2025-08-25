import h5py
import numpy as np
from tqdm import tqdm


class H5RadioMapDataLoader:
    def __init__(self, h5_path, phase='train', batch_size=64):
        """
        初始化H5文件数据加载器

        参数:
            h5_path: H5文件路径
            phase: 数据阶段 'train', 'val' 或 'test'
            batch_size: 批次大小
        """
        self.h5_path = h5_path
        self.phase = phase
        self.batch_size = batch_size
        self.h5_file = None
        self.dataset = None
        self.labels = None
        self.total_samples = 0

    def __enter__(self):
        """上下文管理器入口 - 打开H5文件"""
        self.h5_file = h5py.File(self.h5_path, 'r')
        if self.phase not in self.h5_file:
            raise ValueError(f"Phase '{self.phase}' not found in H5 file")

        phase_group = self.h5_file[self.phase]
        self.dataset = phase_group['inputs']
        self.labels = phase_group['image_gain']
        self.total_samples = phase_group.attrs['total_samples']

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 - 关闭H5文件"""
        if self.h5_file:
            self.h5_file.close()

    def get_sample(self, index):
        """获取单个样本"""
        if index >= self.total_samples:
            raise IndexError(f"Index {index} out of range (total samples: {self.total_samples})")

        return np.array(self.dataset[index]), np.array(self.labels[index])

    def iterate_batches(self, shuffle=False):
        """迭代批次数据 - 使用顺序索引避免错误"""
        # 创建顺序索引
        indices = np.arange(self.total_samples)

        # 如果需要打乱，创建一个打乱的索引列表，但按顺序读取
        if shuffle:
            np.random.shuffle(indices)

        # 按顺序读取批次
        for start_idx in range(0, self.total_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.total_samples)

            # 读取当前批次的数据
            batch_data = []
            batch_labels = []

            for i in range(start_idx, end_idx):
                # 使用实际索引或打乱后的索引
                actual_idx = indices[i] if shuffle else i
                data, label = self.get_sample(actual_idx)
                batch_data.append(data)
                batch_labels.append(label)

            # 转换为numpy数组
            batch_data = np.array(batch_data)
            batch_labels = np.array(batch_labels)

            yield batch_data, batch_labels

    def get_all_data(self):
        """获取所有数据（注意：可能占用大量内存）"""
        all_data = []
        all_labels = []

        for i in tqdm(range(self.total_samples), desc="读取所有数据"):
            data, label = self.get_sample(i)
            all_data.append(data)
            all_labels.append(label)

        return np.array(all_data), np.array(all_labels)


def read_h5_file(h5_path):
    """
    读取H5文件并显示基本信息

    参数:
        h5_path: H5文件路径
    """
    print(f"读取H5文件: {h5_path}")
    print("=" * 50)

    try:
        with h5py.File(h5_path, 'r') as hf:
            # 显示文件属性
            print("文件属性:")
            for key, value in hf.attrs.items():
                print(f"  {key}: {value}")
            print()

            # 显示各个阶段的数据信息
            for phase in ['train', 'val', 'test']:
                if phase in hf:
                    group = hf[phase]
                    print(f"{phase}阶段:")
                    print(f"  样本数量: {group.attrs.get('total_samples', '未知')}")

                    if 'inputs' in group:
                        print(f"  输入数据形状: {group['inputs'].shape}")
                        print(f"  输入数据类型: {group['inputs'].dtype}")

                    if 'image_gain' in group:
                        print(f"  图像增益数据形状: {group['image_gain'].shape}")
                        print(f"  图像增益数据类型: {group['image_gain'].dtype}")

                    print()
    except Exception as e:
        print(f"读取H5文件时出错: {e}")
        print("文件可能已损坏或格式不正确")


def example_usage(h5_path):
    """
    示例用法展示

    参数:
        h5_path: H5文件路径
    """
    # 1. 读取文件基本信息
    read_h5_file(h5_path)

    # 2. 使用数据加载器
    try:
        with H5RadioMapDataLoader(h5_path, phase='train', batch_size=32) as loader:
            print(f"总样本数: {loader.total_samples}")

            # 获取单个样本
            sample_data, sample_label = loader.get_sample(0)
            print(f"单个样本数据形状: {sample_data.shape}")
            print(f"单个样本标签形状: {sample_label.shape}")

            # 迭代批次
            print("\n迭代批次数据:")
            for batch_idx, (batch_data, batch_labels) in enumerate(loader.iterate_batches(shuffle=False)):
                print(f"批次 {batch_idx}: 数据形状={batch_data.shape}, 标签形状={batch_labels.shape}")
                if batch_idx >= 2:  # 只显示前3个批次作为示例
                    break
    except Exception as e:
        print(f"使用数据加载器时出错: {e}")


if __name__ == "__main__":
    h5_path = r"/home/data/path_loss_data/RadioSeer/radiomap_data.h5"

    try:
        example_usage(h5_path)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        print("请检查文件路径是否正确以及文件是否完整")

    # 尝试使用更简单的方法读取文件
    print("\n尝试简单读取文件内容:")
    try:
        with h5py.File(h5_path, 'r') as f:
            print("文件中的键:", list(f.keys()))
            for key in f.keys():
                print(f"组 '{key}' 中的键:", list(f[key].keys()))
    except Exception as e:
        print(f"简单读取时出错: {e}")