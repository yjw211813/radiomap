
from __future__ import print_function, division
import os
from csv import excel

import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

# dir_gainDPM="gain/DPM/",
# dir_gainDPMcars="gain/carsDPM/",
# dir_gainIRT2="gain/IRT2/",
# dir_gainIRT2cars="gain/carsIRT2/",
# dir_buildings="png/",
# dir_antenna= ,

# 调试目录

class RadioMapSeerLoader(Dataset):
    def __init__(self,
                 simuSetDict,
                 maps_inds=np.zeros(1),  # 可选的地图索引序列，默认为0（使用标准划分）
                 phase="train",  # 数据集阶段："train", "val", "test", "custom"
                 transform=transforms.ToTensor()):

        self.ind1 = 0  # 起始索引
        self.ind2 = 0  # 末尾索引
        self.dir_dataset = r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/"  # 数据集文件夹
        self.numTx = 80  # 信源数量设定
        self.thresh = 0.05  # 环境噪声
        self.simulation = "rand"  # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
        self.carsSimul = "yes"  # 是否开启小车作为仿真
        self.carsInput = "yes"  # 是否将小车图作为模型输入
        self.IRT2maxW = 0.3  # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
        self.cityMap = "complete"  # 是否输入完全的城市地图
        self.missing = 1  # 地图缺失号码
        self.fix_samples = 300  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
        self.num_samples_low = 10  # 最低采样数
        self.num_samples_high = 300  # 最高采样数
        self.inter_flag = True  # 看是否需要插值图像
        self.scale256_flag = False  # 取值范围是否为0 - 255
        self.sample_flag = True  # 是否有采样输入
        self.loss_samples_flag = False  # 是否定义loss为稀疏采样loss
        self.formula_flag = False

        self.scale256_flag = False
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

        self.arr = np.arange(256)
        self.one = np.ones(256)
        self.img_temp = np.outer(self.arr,self.one)

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
        if self.scale256_flag:
            return image_gain * 256  # 统一缩放
        else:
            return image_gain

    def _load_buildings_map(self, map_name):
        """加载建筑物地图"""
        if self.cityMap == "complete":
            img_path = os.path.join(self.dir_buildings, map_name)
            if self.scale256_flag:
                return io.imread(img_path)
            else:
                return io.imread(img_path) / 256

        # 处理缺失建筑物的情况
        missing_val = np.random.randint(1, 5) if self.cityMap == "rand" else self.missing
        version = np.random.randint(1, 7)
        dir_path = os.path.join(self.dir_buildings+str(missing_val), str(version))
        img_path = os.path.join(dir_path, map_name)
        if self.scale256_flag:
            return io.imread(img_path)
        else:
            return io.imread(img_path) / 256

    def _load_transmitter_map(self, source_name):
        """加载发射器位置图"""
        img_path = os.path.join(self.dir_Tx, source_name)
        if self.scale256_flag:
            return io.imread(img_path)
        else:
            return io.imread(img_path) / 256

    def create_input_samples(self, image_gain):
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

    def idw_interpolate_sample(self,img_sample, k=5, power=2):
        """
           NumPy版本的反距离加权插值
           :param matrix: 输入数组，形状(H, W)
           :param k: 使用的最近邻点数量
           :param power: 距离权重指数
           :return: 插值后的数组，形状与输入相同
           """
        # 创建输入数据的副本，避免修改原始数据
        data = np.copy(img_sample)
        height, width = data.shape

        # 获取所有非零点的坐标和值
        non_zero_mask = data != 0
        non_zero_coords = np.argwhere(non_zero_mask)
        non_zero_values = data[non_zero_mask]

        # 如果没有非零点，直接返回副本
        if len(non_zero_coords) == 0:
            return data

        # 构建KDTree加速最近邻搜索
        tree = cKDTree(non_zero_coords)

        # 获取所有零值点坐标
        zero_coords = np.argwhere(data == 0)

        # 批量查询所有零值点的k个最近邻
        if len(zero_coords) > 0:  # 确保有需要插值的点
            distances, indices = tree.query(zero_coords, k=k)

            # 避免除以零错误
            distances = np.maximum(distances, 1e-12)

            # 计算权重 (1/d^power)
            weights = 1 / (distances ** power)

            # 获取对应的非零值

            neighbor_values = non_zero_values[indices]


            # 计算加权平均值
            weighted_sum = np.sum(weights * neighbor_values, axis=1)
            sum_weights = np.sum(weights, axis=1)
            interpolated_values = weighted_sum / sum_weights

            # 更新零值点
            for (y, x), value in zip(zero_coords, interpolated_values):
                data[y, x] = value

        return data

    def _load_cars_map(self, map_name):
        """加载车辆地图"""
        img_path = os.path.join(self.dir_cars, map_name)
        if self.scale256_flag:
            return io.imread(img_path)
        else:
            return io.imread(img_path) / 256

    def _get_loss_samples(self):

        loss_samples = np.zeros((self.height, self.width))

        # 确定采样点数
        if self.fix_samples == 0:  # 随机采样点数
            num_samples = np.random.randint(self.num_samples_low, self.num_samples_high)
        else:  # 固定采样点数
            num_samples = int(self.fix_samples)

        # 生成随机采样点
        x_samples = np.random.randint(0, self.height, size=num_samples)
        y_samples = np.random.randint(0, self.width, size=num_samples)

        loss_samples[x_samples, y_samples] = 1

        return loss_samples

    def fusing_building(self,interpolate_data,image_buildings):
        mask = image_buildings > 0
        interpolate_data[mask] = 0

        return interpolate_data

    def fusing_cars(self,interpolate_data,image_cars):
        mask = image_cars > 0
        interpolate_data[mask] = interpolate_data[mask]/2

        return interpolate_data

    def objective(self, x, theta, c):
        return c - 10 * theta * x

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

        if self.sample_flag == True:
            input_samples = self.create_input_samples(image_gain)
            if self.formula_flag == True:

                xk, yk = np.where(input_samples != 0)
                xk, yk = xk.reshape(xk.shape[0], 1), yk.reshape(yk.shape[0], 1)
                p, q = np.where(image_Tx != 0)

                x = np.log10(np.sqrt(np.square(xk - p) + np.square(yk - q)) + 1e-30).flatten()
                y = input_samples[xk, yk].flatten()

                pop, _ = curve_fit(self.objective, x, y)
                theta, c = pop
                genImg = c - 10 * theta * np.log10(np.sqrt(np.square(p - self.img_temp) + np.square(q - self.img_temp.T)) + 1e-30)



            if self.inter_flag == True:

                interpolate_data = self.idw_interpolate_sample(input_samples, k=5)
                interpolate_data =self.fusing_building(interpolate_data,image_buildings)

                input_layers = [image_buildings, image_Tx, input_samples, genImg, interpolate_data]
            else:
                input_layers = [image_buildings, image_Tx, input_samples, genImg]


        else:
            input_layers = [image_buildings, image_Tx]
        # 添加车辆通道（如果需要）
        if self.carsInput != "no":
            image_cars = self._load_cars_map(map_name)
            input_layers.append(image_cars)
            if self.sample_flag == True and self.inter_flag == True:
                input_layers[-2] = self.fusing_cars(input_layers[-2],image_cars)





        inputs = np.stack(input_layers, axis=2)
        # 应用数据转换
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
        if self.loss_samples_flag:
            loss_samples = self._get_loss_samples()
            if self.transform:
                loss_samples = self.transform(loss_samples).type(torch.float32)
            return inputs, image_gain ,loss_samples

        return inputs, image_gain
