import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RadioMapSeerH5Loader(Dataset):
    def __init__(self, h5_path, phase='train'):
        """
        初始化H5文件数据集

        参数:
            h5_path: H5文件路径
            phase: 数据阶段 'train', 'val' 或 'test'
        """
        self.h5_path = h5_path
        self.phase = phase
        self.h5_file = None
        self.inputs_dataset = None
        self.labels_dataset = None
        self.total_samples = 0

        # 立即打开文件并获取基本信息
        self._open_file()

    def _open_file(self):
        """打开H5文件并初始化数据集"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            if self.phase not in self.h5_file:
                raise ValueError(f"Phase '{self.phase}' not found in H5 file")

            phase_group = self.h5_file[self.phase]
            self.inputs_dataset = phase_group['inputs']
            self.labels_dataset = phase_group['image_gain']
            self.total_samples = phase_group.attrs['total_samples']

    def __len__(self):
        """返回数据集大小"""
        return self.total_samples

    def __getitem__(self, index):
        """获取单个样本"""
        if index >= self.total_samples:
            raise IndexError(f"Index {index} out of range (total samples: {self.total_samples})")

        # 确保文件已打开
        if self.h5_file is None:
            self._open_file()

        # 读取数据并转换为torch张量
        inputs = torch.from_numpy(np.array(self.inputs_dataset[index]))
        labels = torch.from_numpy(np.array(self.labels_dataset[index]))

        return inputs, labels

    def __del__(self):
        """析构函数，确保文件被关闭"""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None


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


def create_dataloaders(h5_path, train_batch_size=64, val_batch_size=64, test_batch_size=64, num_workers=4):
    """
    创建训练、验证和测试数据加载器

    参数:
        h5_path: H5文件路径
        train_batch_size: 训练批次大小
        val_batch_size: 验证批次大小
        test_batch_size: 测试批次大小
        num_workers: 数据加载工作进程数

    返回:
        包含train, val, test数据加载器的字典
    """
    # 创建数据集
    Radio_train = RadioMapSeerH5Loader(h5_path, phase="train")
    Radio_val = RadioMapSeerH5Loader(h5_path, phase="val")
    Radio_test = RadioMapSeerH5Loader(h5_path, phase="test")

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=train_batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(Radio_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(Radio_test, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    }

    return dataloaders