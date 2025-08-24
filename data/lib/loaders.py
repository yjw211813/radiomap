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
        input_samples = self.create_input_samples(image_gain)

        if self.inter_flag == True:
            interpolate_data = self.idw_interpolate_sample(input_samples, k=5)
            input_layers = [image_buildings, image_Tx, input_samples, interpolate_data]
        else:
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


class RadioUNet_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="/home/data/path_loss_data/RadioSeer/RadioMapSeer/",
                 numTx=80,
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:
            inputs: The RadioUNet inputs.
            image_gain

        """

        # self.phase=phase

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh

        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        if simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/DPM/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsDPM/"
        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsIRT2/"
        elif simulation == "rand":
            if carsSimul == "no":
                self.dir_gainDPM = self.dir_dataset + "gain/DPM/"
                self.dir_gainIRT2 = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gainDPM = self.dir_dataset + "gain/carsDPM/"
                self.dir_gainIRT2 = self.dir_dataset + "gain/carsIRT2/"

        self.IRT2maxW = IRT2maxW

        self.cityMap = cityMap
        self.missing = missing
        if cityMap == "complete":
            self.dir_buildings = self.dir_dataset + "png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset + "png/buildings_missing"  # a random index will be concatenated in the code
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        self.transform = transform

        self.dir_Tx = self.dir_dataset + "png/antennas/"
        # later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput != "no":
            self.dir_cars = self.dir_dataset + "png/cars/"

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings + str(self.missing) + "/" + str(version) + "/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))

        # Load radio map:
        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 255
        else:  # random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            # image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w = np.random.uniform(0, self.IRT2maxW)  # IRT2 weight of random average
            image_gain = w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256 \
                         + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        # inputs to radioUNet
        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx], axis=2)
            # The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence,
            # so we can use the same learning rate as RadioUNets
        else:  # cars
            # Normalization, so all settings can have the same learning rate
            image_buildings = image_buildings / 256
            image_Tx = image_Tx / 256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 256
            inputs = np.stack([image_buildings, image_Tx, image_cars], axis=2)
            # note that ToTensor moves the channel from the last asix to the first!

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            # note that ToTensor moves the channel from the last asix to the first!

        return [inputs, image_gain]


class RadioUNet_c_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="/home/data/path_loss_data/RadioSeer/RadioMapSeer/",
                 numTx=2,
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="yes",
                 carsInput="yes",
                 cityMap="complete",
                 missing=1,
                 num_samples=300,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            num_samples: number of samples in the sparse IRT4 radio map. Default=300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:

        """
        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh

        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        if simulation == "IRT4":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/IRT4/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsIRT4/"

        elif simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/DPM/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsDPM/"
        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsIRT2/"

        self.cityMap = cityMap
        self.missing = missing
        if cityMap == "complete":
            self.dir_buildings = self.dir_dataset + "png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset + "png/buildings_missing"  # a random index will be concatenated in the code
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        self.transform = transform

        self.num_samples = num_samples

        self.dir_Tx = self.dir_dataset + "png/antennas/"
        # later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput != "no":
            self.dir_cars = self.dir_dataset + "png/cars/"

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings + str(self.missing) + "/" + str(version) + "/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))

        # Load radio map:
        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 256
        else:  # random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            # image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w = np.random.uniform(0, self.IRT2maxW)  # IRT2 weight of random average
            image_gain = w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256 \
                         + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        # Saprse IRT4 samples, determenistic and fixed samples per map
        image_samples = np.zeros((self.width, self.height))
        seed_map = np.sum(image_buildings)  # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)
        x_samples = np.random.randint(0, 255, size=self.num_samples)
        y_samples = np.random.randint(0, 255, size=self.num_samples)
        image_samples[x_samples, y_samples] = 1

        # inputs to radioUNet
        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx], axis=2)
            # The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence,
            # so we can use the same learning rate as RadioUNets
        else:  # cars
            # Normalization, so all settings can have the same learning rate
            image_buildings = image_buildings / 256
            image_Tx = image_Tx / 256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 256
            inputs = np.stack([image_buildings, image_Tx, image_cars], axis=2)
            # note that ToTensor moves the channel from the last asix to the first!

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_samples = self.transform(image_samples).type(torch.float32)

        return [inputs, image_gain, image_samples]


class RadioUNet_s(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="/home/data/path_loss_data/RadioSeer/RadioMapSeer/",
                 numTx=80,
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 fix_samples=0,
                 num_samples_low=10,
                 num_samples_high=300,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10.
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:
            inputs: The RadioUNet inputs.
            image_gain

        """

        # self.phase=phase

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh

        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        if simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/DPM/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsDPM/"
        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsIRT2/"
        elif simulation == "rand":
            if carsSimul == "no":
                self.dir_gainDPM = self.dir_dataset + "gain/DPM/"
                self.dir_gainIRT2 = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gainDPM = self.dir_dataset + "gain/carsDPM/"
                self.dir_gainIRT2 = self.dir_dataset + "gain/carsIRT2/"

        self.IRT2maxW = IRT2maxW

        self.cityMap = cityMap
        self.missing = missing
        if cityMap == "complete":
            self.dir_buildings = self.dir_dataset + "png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset + "png/buildings_missing"  # a random index will be concatenated in the code
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        self.fix_samples = fix_samples
        self.num_samples_low = num_samples_low
        self.num_samples_high = num_samples_high

        self.transform = transform

        self.dir_Tx = self.dir_dataset + "png/antennas/"
        # later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput != "no":
            self.dir_cars = self.dir_dataset + "png/cars/"

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings + str(self.missing) + "/" + str(version) + "/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings)) / 256

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx)) / 256

        # Load radio map:
        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 256
        else:  # random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            # image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w = np.random.uniform(0, self.IRT2maxW)  # IRT2 weight of random average
            image_gain = w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256 \
                         + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        image_gain = image_gain * 256  # we use this normalization so all RadioUNet methods can have the same learning rate.
        # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
        # Important: when evaluating the accuracy, remember to devide the errors by 256!

        # input measurements
        image_samples = np.zeros((256, 256))
        if self.fix_samples == 0:
            num_samples = np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_samples = np.floor(self.fix_samples).astype(int)
        x_samples = np.random.randint(0, 255, size=num_samples)
        y_samples = np.random.randint(0, 255, size=num_samples)
        image_samples[x_samples, y_samples] = image_gain[x_samples, y_samples, 0]

        # inputs to radioUNet
        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx, image_samples], axis=2)
            # The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence,
            # so we can use the same learning rate as RadioUNets
        else:  # cars
            # Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 256
            inputs = np.stack([image_buildings, image_Tx, image_samples, image_cars], axis=2)
            # note that ToTensor moves the channel from the last asix to the first!

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            # note that ToTensor moves the channel from the last asix to the first!

        return [inputs, image_gain]


class RadioUNet_s_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="/home/data/path_loss_data/RadioSeer/RadioMapSeer/",
                 numTx=2,
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=1,
                 data_samples=300,
                 fix_samples=0,
                 num_samples_low=10,
                 num_samples_high=299,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            data_samples: number of samples in the sparse IRT4 radio map. Default=300. All input samples are taken from the data_samples
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10.
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:

        """
        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh

        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        if simulation == "IRT4":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/IRT4/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsIRT4/"

        elif simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/DPM/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsDPM/"
        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsIRT2/"

        self.cityMap = cityMap
        self.missing = missing
        if cityMap == "complete":
            self.dir_buildings = self.dir_dataset + "png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset + "png/buildings_missing"  # a random index will be concatenated in the code
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        self.data_samples = data_samples
        self.fix_samples = fix_samples
        self.num_samples_low = num_samples_low
        self.num_samples_high = num_samples_high

        self.transform = transform

        self.dir_Tx = self.dir_dataset + "png/antennas/"
        # later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput != "no":
            self.dir_cars = self.dir_dataset + "png/cars/"

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings + str(self.missing) + "/" + str(version) + "/", name1)
            str(self.missing)
        image_buildings = np.asarray(
            io.imread(img_name_buildings))  # Will be normalized later, after random seed is computed from it

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx)) / 256

        # Load radio map:
        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 256
        else:  # random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            # image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w = np.random.uniform(0, self.IRT2maxW)  # IRT2 weight of random average
            image_gain = w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256 \
                         + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        image_gain = image_gain * 256  # we use this normalization so all RadioUNet methods can have the same learning rate.
        # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
        # Important: when evaluating the accuracy, remember to devide the errors by 256!

        # Saprse IRT4 samples, determenistic and fixed samples per map
        sparse_samples = np.zeros((self.width, self.height))
        seed_map = np.sum(image_buildings)  # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)
        x_samples = np.random.randint(0, 255, size=self.data_samples)
        y_samples = np.random.randint(0, 255, size=self.data_samples)
        sparse_samples[x_samples, y_samples] = 1

        # input samples from the sparse gain samples
        input_samples = np.zeros((256, 256))
        if self.fix_samples == 0:
            num_in_samples = np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_in_samples = np.floor(self.fix_samples).astype(int)

        data_inds = range(self.data_samples)
        input_inds = np.random.permutation(data_inds)[0:num_in_samples[0]]
        x_samples_in = x_samples[input_inds]
        y_samples_in = y_samples[input_inds]
        input_samples[x_samples_in, y_samples_in] = image_gain[x_samples_in, y_samples_in, 0]

        # normalize image_buildings, after random seed computed from it as an int
        image_buildings = image_buildings / 256

        # inputs to radioUNet
        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx, input_samples], axis=2)
            # The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence,
            # so we can use the same learning rate as RadioUNets
        else:  # cars
            # Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 256
            inputs = np.stack([image_buildings, image_Tx, input_samples, image_cars], axis=2)
            # note that ToTensor moves the channel from the last asix to the first!

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            sparse_samples = self.transform(sparse_samples).type(torch.float32)

        return [inputs, image_gain, sparse_samples]













    