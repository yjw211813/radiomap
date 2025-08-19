import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from data.lib.loaders import RadioMapSeerLoader
simuSetDict = {
    "ind1": 0,  # 起始索引
    "ind2": 0,  # 末尾索引
    "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",  # 数据集文件夹
    "numTx": 80,  # 信源数量设定
    "thresh": 0.05,  # 环境噪声
    "simulation": "rand",  # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
    "carsSimul": "yes",  # 是否开启小车作为仿真
    "carsInput": "yes",  # 是否将小车图作为模型输入
    "IRT2maxW": 0.3,  # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
    "cityMap": "complete",  # 是否输入完全的城市地图
    "missing": 1,  # 地图缺失号码
    "fix_samples": 300,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
    "num_samples_low": 10,  # 最低采样数
    "num_samples_high": 300  # 最高采样数
}

def save_datasets_to_hdf5(simuSetDict, hdf5_path, train_batch_size=64, test_batch_size=64, num_workers=4):
    # 创建数据集实例
    datasets = {
        'train': RadioMapSeerLoader(simuSetDict, phase="train"),
        'val': RadioMapSeerLoader(simuSetDict, phase="val"),
        'test': RadioMapSeerLoader(simuSetDict, phase="test")
    }

    # 创建HDF5文件
    with h5py.File(hdf5_path, 'w') as hf:
        for phase in ['train', 'val', 'test']:
            ds = datasets[phase]
            total_samples = len(ds)

            # 获取样本形状和数据类型
            sample_inputs, sample_image_gain = ds[0]
            inputs_shape = (total_samples,) + sample_inputs.shape
            image_gain_shape = (total_samples,) + sample_image_gain.shape

            # 预分配HDF5数据集
            grp = hf.create_group(phase)
            inputs_dset = grp.create_dataset(
                'inputs',
                shape=inputs_shape,
                dtype=sample_inputs.dtype,
                compression='gzip',
                chunks=(1,) + sample_inputs.shape
            )
            image_gain_dset = grp.create_dataset(
                'image_gain',
                shape=image_gain_shape,
                dtype=sample_image_gain.dtype,
                compression='gzip',
                chunks=(1,) + sample_image_gain.shape
            )

            # 使用多线程读取并写入数据
            batch_size = train_batch_size if phase == 'train' else test_batch_size
            indices = list(range(total_samples))

            # 线程池处理函数
            def process_batch(batch_indices):
                batch_inputs = []
                batch_image_gains = []
                for idx in batch_indices:
                    inputs, image_gain = ds[idx]
                    batch_inputs.append(inputs)
                    batch_image_gains.append(image_gain)
                return batch_indices, np.stack(batch_inputs), np.stack(batch_image_gains)

            # 分批处理索引
            batches = [
                indices[i:i + batch_size]
                for i in range(0, total_samples, batch_size)
            ]

            # 使用线程池并行处理
            with ThreadPool(processes=num_workers) as pool:
                for batch_indices, batch_inputs, batch_image_gains in tqdm(
                        pool.imap(process_batch, batches),
                        total=len(batches),
                        desc=f"Processing {phase} set"
                ):
                    start_idx = batch_indices[0]
                    end_idx = start_idx + len(batch_indices)
                    inputs_dset[start_idx:end_idx] = batch_inputs
                    image_gain_dset[start_idx:end_idx] = batch_image_gains


# 使用示例
if __name__ == "__main__":

    save_datasets_to_hdf5(
        simuSetDict,
        hdf5_path="/home/data/path_loss_data/radiomap_data.h5",
        train_batch_size=64,
        test_batch_size=64,
        num_workers=8  # 根据CPU核心数调整
    )