from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")


 #dir_gainDPM="gain/DPM/",
 #dir_gainDPMcars="gain/carsDPM/",
 #dir_gainIRT2="gain/IRT2/",
 #dir_gainIRT2cars="gain/carsIRT2/",
 #dir_buildings="png/",
 #dir_antenna= ,
                    
# 调试目录

class RadioMapSeerLoader(Dataset):
    def __init__(self,
                 simuSetDict,
                 maps_inds=np.zeros(1),  # 可选的地图索引序列，默认为0（使用标准划分）
                 phase="train",  # 数据集阶段："train", "val", "test", "custom"
                 transform=transforms.ToTensor()):



        # 将设置字典中的参数转为类属性
        for key, value in simuSetDict.items():
            setattr(self, key, value)

        # 数据集对象初始化
        self.simuSetDict = simuSetDict  # 保留设置字典
        self._init_index(maps_inds, phase)
        self._setup_directories()  # 统一设置所有目录
        self.transform = transform  # 数据预处理方式

        if self.simulation == "IRT4" and self.numTx > 2:
            self.numTx = 2

        self.height = 256
        self.width = 256

    def _init_index(self, maps_inds, phase):
        """初始化地图索引和数据集范围"""
        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1, self.ind2 = 0, 500
        elif phase == "val":
            self.ind1, self.ind2 = 501, 600
        elif phase == "test":
            self.ind1, self.ind2 = 601, 699
        else:  # custom range
            self.ind1 = self.ind1 if hasattr(self, 'ind1') else self.simuSetDict.get("ind1", 0)
            self.ind2 = self.ind2 if hasattr(self, 'ind2') else self.simuSetDict.get("ind2", 0)

    def _setup_directories(self):
        """统一设置所有需要的目录路径"""
        # 增益图目录
        if self.simulation == "IRT4":
            base = "carsIRT4/" if self.carsSimul == "yes" else "IRT4/"
            self.dir_gain = os.path.join(self.dir_dataset, "gain", base)
        elif self.simulation == "DPM":
            base = "carsDPM/" if self.carsSimul == "yes" else "DPM/"
            self.dir_gain = os.path.join(self.dir_dataset, "gain", base)
        elif self.simulation == "IRT2":
            base = "carsIRT2/" if self.carsSimul == "yes" else "IRT2/"
            self.dir_gain = os.path.join(self.dir_dataset, "gain", base)

        # 为随机模拟模式准备备用目录
        if self.simulation == "rand" or not hasattr(self, 'dir_gain'):
            base = "cars" if self.carsSimul == "yes" else ""
            self.dir_gainDPM = os.path.join(self.dir_dataset, "gain", f"{base}DPM/")
            self.dir_gainIRT2 = os.path.join(self.dir_dataset, "gain", f"{base}IRT2/")

        # 建筑物目录
        if self.cityMap == "complete":
            self.dir_buildings = os.path.join(self.dir_dataset, "png", "buildings_complete/")
        else:
            self.dir_buildings = os.path.join(self.dir_dataset, "png", "buildings_missing")

        # 发射器和车辆目录
        self.dir_Tx = os.path.join(self.dir_dataset, "png", "antennas/")
        if self.carsInput != "no":
            self.dir_cars = os.path.join(self.dir_dataset, "png", "cars/")

    def _get_map_and_source_index(self, index):
        """计算地图索引和发射器索引"""
        idxr = np.floor(index / self.numTx).astype(int)  # 地图索引
        idxc = index - idxr * self.numTx  # 发射器索引
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1  # 实际地图编号
        return dataset_map_ind, idxc

    def _load_gain_map(self, source_name):
        """加载并处理增益图"""
        if self.simulation != "rand":
            img_path = os.path.join(self.dir_gain, source_name)
            image_gain = np.expand_dims(io.imread(img_path), axis=2) / 256.0
        else:
            # 随机混合DPM和IRT2
            img_path_dpm = os.path.join(self.dir_gainDPM, source_name)
            img_path_irt2 = os.path.join(self.dir_gainIRT2, source_name)
            w = np.random.uniform(0, self.IRT2maxW)
            gain_dpm = np.expand_dims(io.imread(img_path_dpm), axis=2) / 256.0
            gain_irt2 = np.expand_dims(io.imread(img_path_irt2), axis=2) / 256.0
            image_gain = w * gain_irt2 + (1 - w) * gain_dpm

        # 路径损耗阈值处理
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = (image_gain - self.thresh) / (1 - self.thresh)

        return image_gain * 256  # 统一缩放

    def _load_buildings_map(self, map_name):
        """加载建筑物地图"""
        if self.cityMap == "complete":
            img_path = os.path.join(self.dir_buildings, map_name)
            return io.imread(img_path) / 256.0

        # 处理缺失建筑物的情况
        missing_val = np.random.randint(1, 5) if self.cityMap == "rand" else self.missing
        version = np.random.randint(1, 7)
        dir_path = os.path.join(self.dir_buildings+str(missing_val), str(version))
        img_path = os.path.join(dir_path, map_name)
        return io.imread(img_path)

    def _load_transmitter_map(self, source_name):
        """加载发射器位置图"""
        img_path = os.path.join(self.dir_Tx, source_name)
        return io.imread(img_path)

    def _create_input_samples(self, image_gain):
        """创建输入采样点图"""
        image_samples = np.zeros((self.height, self.width))

        # 确定采样点数
        if self.fix_samples == 0:  # 随机采样点数
            num_samples = np.random.randint(self.num_samples_low, self.num_samples_high)
        else:  # 固定采样点数
            num_samples = int(self.fix_samples)

        # 生成随机采样点
        x_samples = np.random.randint(0, self.height, size=num_samples)
        y_samples = np.random.randint(0, self.width, size=num_samples)

        # 填充增益值
        image_samples[x_samples, y_samples] = image_gain[x_samples, y_samples, 0]

        return image_samples

    def _load_cars_map(self, map_name):
        """加载车辆地图"""
        img_path = os.path.join(self.dir_cars, map_name)
        return io.imread(img_path) 

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取地图和发射器信息
        map_idx, source_idx = self._get_map_and_source_index(idx)
        map_name = f"{map_idx}.png"
        source_name = f"{map_idx}_{source_idx}.png"

        # 加载主要数据
        image_gain = self._load_gain_map(source_name)
        image_buildings = self._load_buildings_map(map_name)
        image_Tx = self._load_transmitter_map(source_name)
        input_samples = self._create_input_samples(image_gain)

        # 构建输入张量
        input_layers = [image_buildings, image_Tx, input_samples]

        # 添加车辆通道（如果需要）
        if self.carsInput != "no":
            image_cars = self._load_cars_map(map_name)
            input_layers.append(image_cars)

        inputs = np.stack(input_layers, axis=2)

        # 应用数据转换
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)

        return inputs, image_gain



    