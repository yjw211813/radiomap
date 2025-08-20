import h5py
import numpy as np
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from data.lib.loaders import RadioMapSeerLoader
import torch  # 确保导入torch

simuSetDict = {
    "ind1": 0,  # 起始索引
    "ind2": 0,  # 末尾索引
    "dir_dataset": r"C:\Users\Administrator\Desktop\bin\pystft\RadioMapSeer/",  # 数据集文件夹
    "numTx": 80,  # 信源数量设定
    "thresh": 0.05,  # 环境噪声
    "simulation": "rand",  # 模拟类型
    "carsSimul": "yes",  # 是否开启小车作为仿真
    "carsInput": "yes",  # 是否将小车图作为模型输入
    "IRT2maxW": 0.3,  # IRT2的最大加权值
    "cityMap": "complete",  # 是否输入完全的城市地图
    "missing": 1,  # 地图缺失号码
    "fix_samples": 300,  # 采样数量
    "num_samples_low": 10,  # 最低采样数
    "num_samples_high": 300,  # 最高采样数
    "inter_flag": True  # 看是否需要插值图像
}

def save_datasets_to_hdf5(simuSetDict, hdf5_path, train_batch_size=64, test_batch_size=64, num_workers=12):
    datasets = {
        'train': RadioMapSeerLoader(simuSetDict, phase="train"),
        'val': RadioMapSeerLoader(simuSetDict, phase="val"),
        'test': RadioMapSeerLoader(simuSetDict, phase="test")
    }

    with h5py.File(hdf5_path, 'w') as hf:
        hf.attrs['description'] = "RadioMapSeer Dataset"
        hf.attrs['version'] = "1.0"
        hf.attrs['author'] = "Your Name"

        phase_names = ['train', 'val', 'test']
        for phase in tqdm(phase_names, desc="Processing datasets"):
            ds = datasets[phase]
            total_samples = len(ds)
            grp = hf.create_group(phase)
            grp.attrs['total_samples'] = total_samples
            grp.attrs['phase'] = phase

            # 获取样本并转换为NumPy数组
            sample_inputs, sample_image_gain = ds[0]
            # 将PyTorch张量转为NumPy数组
            if isinstance(sample_inputs, torch.Tensor):
                sample_inputs = sample_inputs.numpy()
            if isinstance(sample_image_gain, torch.Tensor):
                sample_image_gain = sample_image_gain.numpy()

            inputs_shape = (total_samples,) + sample_inputs.shape
            image_gain_shape = (total_samples,) + sample_image_gain.shape

            # 使用NumPy数据类型创建数据集
            inputs_dset = grp.create_dataset(
                'inputs',
                shape=inputs_shape,
                dtype=sample_inputs.dtype,  # 使用NumPy dtype
                compression='gzip',
                chunks=(1,) + sample_inputs.shape
            )
            image_gain_dset = grp.create_dataset(
                'image_gain',
                shape=image_gain_shape,
                dtype=sample_image_gain.dtype,  # 使用NumPy dtype
                compression='gzip',
                chunks=(1,) + sample_image_gain.shape
            )

            batch_size = train_batch_size if phase == 'train' else test_batch_size
            with ThreadPool(processes=num_workers) as pool:
                indices = list(range(total_samples))
                batches = [
                    indices[i:i + batch_size]
                    for i in range(0, total_samples, batch_size)
                ]

                for batch_idx in tqdm(range(len(batches)), desc=f"Processing {phase} batches", leave=False):
                    batch_indices = batches[batch_idx]
                    batch_inputs = []
                    batch_image_gains = []
                    for idx in batch_indices:
                        inputs, image_gain = ds[idx]
                        # 确保转换为NumPy数组
                        if isinstance(inputs, torch.Tensor):
                            inputs = inputs.numpy()
                        if isinstance(image_gain, torch.Tensor):
                            image_gain = image_gain.numpy()
                        batch_inputs.append(inputs)
                        batch_image_gains.append(image_gain)

                    batch_inputs = np.stack(batch_inputs)
                    batch_image_gains = np.stack(batch_image_gains)

                    start_idx = batch_indices[0]
                    end_idx = start_idx + len(batch_indices)
                    inputs_dset[start_idx:end_idx] = batch_inputs
                    image_gain_dset[start_idx:end_idx] = batch_image_gains

if __name__ == "__main__":
    save_datasets_to_hdf5(
        simuSetDict,
        hdf5_path=r"C:\Users\Administrator\Desktop\bin\pystft/radiomap_data.h5",
        train_batch_size=64,
        test_batch_size=64,
        num_workers=12
    )