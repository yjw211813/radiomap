from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
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
class RadioUNet_c(Dataset):
    # 暂时不太清楚c代表什么含义
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    """RadioMapSeer数据集加载器，用于精确建筑物建模和无测量数据的无线电地图构建（RadioUNet_c）"""
    def __init__(self,maps_inds=np.zeros(1),# 可选的地图索引序列，默认为0（使用标准划分）
                 phase="train",             # 数据集阶段："train", "val", "test", "custom"
                 ind1=0,ind2=0,             # 自定义范围时使用的起始和结束索引
                 dir_dataset=r"/home/data/path_loss_data/RadioSeer/RadioMapSeer",# 数据集根目录
                 numTx=80,                  # 每个地图的发射器数量（最大80）
                 thresh=0.05,               # 路径损耗阈值（0-1），默认0.05
                 simulation="DPM",          # 模拟类型："DPM", "IRT2", "rand"
                 carsSimul="no",            # 是否在模拟中包含车辆："yes"/"no"
                 carsInput="no",            # 输入是否包含车辆通道："yes"/"no"
                 IRT2maxW=1,                # 随机模拟时IRT2的最大权重
                 cityMap="complete",        # 城市地图类型：complete, "missing", "rand"
                 missing=1,                 # 缺失建筑物数量（1-4）
                 transform= transforms.ToTensor()):# 图像转换方法
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
        

        
        #self.phase=phase
        # 初始化地图索引
        if maps_inds.size==1:
            # 默认创建0-699的有序索引
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            # 使用固定随机种子打乱索引（确保可重现性）
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500 # 训练集：前500张地图
        elif phase=="val":
            self.ind1=501
            self.ind2=600 # 验证集：中间100张地图
        elif phase=="test":
            self.ind1=601
            self.ind2=699 # 测试集：最后99张地图
        else:  # custom模式使用自定义范围
            self.ind1=ind1
            self.ind2=ind2
        # 定义根目录下面的文件夹位置
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        # 设置模拟相关路径
        self.simulation=simulation
        # 看是否进行车辆仿真
        self.carsSimul=carsSimul
        # 看是否有car位置作为模型输入
        self.carsInput=carsInput
        # 根据模拟类型确定增益图目录
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand": # 随机混合模式
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        # 设置建筑物地图路径
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"# 缺失建筑物模式
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        # 计算地图索引和发射器索引
        idxr=np.floor(idx/self.numTx).astype(int)               # 地图索引
        idxc=idx-idxr*self.numTx                                # 发射器索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1        # 实际地图编号（+1因为文件从1开始）
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"                   # 地图相关文件
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png" # 发射器相关文件
        
        # 加载建筑物图像
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            # 缺失建筑物模式
            if self.cityMap == "rand":                  # 随机缺失数量
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)    # 随机选择缺失版本
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))

        # 加载发射器位置图
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        # 加载无线电增益图
        # 如果使用rand模式则会进行两种方法的融合
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            # 随机混合DPM和IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average # 随机权重
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # pathloss threshold transform
        # 路径损耗阈值处理
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
                 
        
        # inputs to radioUNet
        # 构建输入数据
        if self.carsInput=="no":
            # 无车辆通道：建筑物+发射器位置
            inputs=np.stack([image_buildings, image_Tx], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!


        return [inputs, image_gain]


class RadioUNet_c_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    """ 可以直接用这个类来进行模型相关训练 """
    """RadioMapSeer数据集加载器，支持稀疏IRT4模拟的无线电地图构建"""

    def __init__(self,
                 maps_inds=np.zeros(1), # 可选的地图索引序列，默认为0（使用标准划分）
                 phase="train",         # 数据集阶段："train", "val", "test", "custom"
                 ind1=0,ind2=0,         # 自定义范围时使用的起始和结束索引
                 dir_dataset=r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",# 数据集根目录
                 numTx=2,               # 每个地图的发射器数量（注意：IRT4最多支持2个发射器）
                 thresh=0.2,            # 路径损耗阈值（0-1），默认0.2
                 simulation="IRT4",     # 模拟类型："IRT4", "DPM", "IRT2"
                 carsSimul="yes",       # 是否在模拟中包含车辆："yes"/"no"
                 carsInput="yes",       # 输入是否包含车辆通道："yes"/"no"
                 cityMap="complete",    # 城市地图类型："complete", "missing", "rand"
                 missing=1,             # 缺失建筑物数量（1-4）
                 num_samples=1000,      # 稀疏采样的样本数量
                 transform= transforms.ToTensor()): # 图像转换方法
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
        # 初始化地图索引
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2

        # 存储基本参数
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh

        # 设置模拟相关路径
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput

        # 根据模拟类型确定增益图目录
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.num_samples=num_samples
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
        
        
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
        
        #Saprse IRT4 samples, determenistic and fixed samples per map
        image_samples = np.zeros((self.width,self.height))
        seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)       
        x_samples=np.random.randint(0, 255, size=self.num_samples)
        y_samples=np.random.randint(0, 255, size=self.num_samples)
        image_samples[x_samples,y_samples]= 1
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_samples = self.transform(image_samples).type(torch.float32)


        return [inputs, image_gain, image_samples]


class RadioUNet_s(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    """RadioMapSeer 数据集加载器，支持稀疏测量输入的无线电地图构建 (RadioUNet_s)"""
    def __init__(self,
                 maps_inds=np.zeros(1), # 可选的地图索引序列，默认为0（使用标准划分）
                 phase="train",         # 数据集阶段："train", "val", "test", "custom"
                 ind1=0,ind2=0, 
                 dir_dataset=r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="DPM",  # 模拟类型："DPM", "IRT2", "rand"
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 fix_samples=0,
                 num_samples_low= 10, 
                 num_samples_high= 300,
                 transform= transforms.ToTensor()):
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
        

        
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
                
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))/256  
        
        #Load Tx (transmitter):
        # 加载并归一化发射器位置图
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
            
        image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                 
        #input measurements
        # === 创建稀疏测量图 ===
        image_samples = np.zeros((256,256)) # 创建全零采样图
        # 确定采样点数
        if self.fix_samples==0:# 随机采样点数
            num_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:# 固定采样点数
            num_samples=np.floor(self.fix_samples).astype(int)               
        x_samples=np.random.randint(0, 255, size=num_samples)
        y_samples=np.random.randint(0, 255, size=num_samples)
        image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]
        
        #inputs to radioUNet
        # 构建输入数据（包含稀疏测量）
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        # 应用数据转换（如转为Tensor）
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!


        return [inputs, image_gain]
    

class RadioUNet_s_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset=r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",
                 numTx=2,                  
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=1,
                 data_samples=300,
                 fix_samples=0,
                 num_samples_low= 10, 
                 num_samples_high= 299,
                 transform= transforms.ToTensor()):
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
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.data_samples=data_samples
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
        
        self.transform= transform
        
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
        
        
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))  #Will be normalized later, after random seed is computed from it
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256 
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
        
        image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                    
        #Saprse IRT4 samples, determenistic and fixed samples per map
        sparse_samples = np.zeros((self.width,self.height))
        seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)       
        x_samples=np.random.randint(0, 255, size=self.data_samples)
        y_samples=np.random.randint(0, 255, size=self.data_samples)
        sparse_samples[x_samples,y_samples]= 1
        
        #input samples from the sparse gain samples
        input_samples = np.zeros((256,256))
        if self.fix_samples==0:
            num_in_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_in_samples=np.floor(self.fix_samples).astype(int)
            
        data_inds=range(self.data_samples)
        input_inds=np.random.permutation(data_inds)[0:num_in_samples[0]]      
        x_samples_in=x_samples[input_inds]
        y_samples_in=y_samples[input_inds]
        input_samples[x_samples_in,y_samples_in]= image_gain[x_samples_in,y_samples_in,0]
        
        #normalize image_buildings, after random seed computed from it as an int
        image_buildings=image_buildings/256
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, input_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, input_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            sparse_samples = self.transform(sparse_samples).type(torch.float32)
            


        return [inputs, image_gain, sparse_samples]
    
"""
simuSetDict = {
    "ind1": 1,
    "ind2": 2,
    "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",
    "numTx": 80,
    "thresh": 0.2,
    "simulation": "DPM",  # 模拟类型："DPM", "IRT2", "rand"
    "carsSimul": "no",
    "carsInput": "no",
    "IRT2maxW": 1,
    "cityMap": "complete",
    "missing": 1,
    "fix_samples": 0,
    "num_samples_low": 10,
    "num_samples_high": 300
}
"""




class RadioMapSeerLoader(Dataset):
    def __init__(self,
                 simuSetDict ,
                 maps_inds=np.zeros(1), # 可选的地图索引序列，默认为0（使用标准划分）
                 phase="train",         # 数据集阶段："train", "val", "test", "custom"
                 transform= transforms.ToTensor()):

        # 数据集对象初始化
        self.simuSetDict = simuSetDict  # 得到设置字典
        self. init_index(maps_inds,phase)
        self.get_dir_gain()     # 得到当前增益路径
        self.get_dir_building() # 得到建筑物的路径
        self.get_dir_Tx()       # 得到发射源位置图像信息的路径
        self.get_dir_cars()     # 得到小车图像信息的路径
        self.transform = transform         # 得到数据预处理方式
        self.height = 256
        self.width = 256




    def init_index(self,maps_inds,phase):
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
            self.ind1 = self.simuSetDict["ind1"]
            self.ind2 = self.simuSetDict["ind2"]

    def get_dir_gain(self):
        if self.simuSetDict["simulation"] == "IRT4":
            if self.simuSetDict["carsSimul"] == "no":
                self.dir_gain = self.simuSetDict["dir_dataset"] + "gain/IRT4/"
            else:
                self.dir_gain = self.simuSetDict["dir_dataset"] + "gain/carsIRT4/"

        elif self.simuSetDict["simulation"]  == "DPM":
            if self.simuSetDict["carsSimul"] == "no":
                self.dir_gain = self.simuSetDict["dir_dataset"] + "gain/DPM/"
            else:
                self.dir_gain = self.simuSetDict["dir_dataset"] + "gain/carsDPM/"
        elif self.simuSetDict["simulation"]  == "IRT2":
            if self.simuSetDict["carsSimul"] == "no":
                self.dir_gain = self.simuSetDict["dir_dataset"] + "gain/IRT2/"
            else:
                self.dir_gain = self.simuSetDict["dir_dataset"] + "gain/carsIRT2/"

        if self.simuSetDict["carsSimul"] == "no":
            self.dir_gainDPM = self.simuSetDict["dir_dataset"] + "gain/DPM/"
            self.dir_gainIRT2 = self.simuSetDict["dir_dataset"] + "gain/IRT2/"
        else:
            self.dir_gainDPM = self.simuSetDict["dir_dataset"] + "gain/carsDPM/"
            self.dir_gainIRT2 = self.simuSetDict["dir_dataset"] + "gain/carsIRT2/"


    def get_dir_building(self):
        if self.simuSetDict["cityMap"]=="complete":
            self.dir_buildings=self.simuSetDict["dir_dataset"]+"png/buildings_complete/" # 缺失建筑物模式
        else:
            self.dir_buildings = self.simuSetDict["dir_dataset"]+"png/buildings_missing" #后续会随机加一个随机丢失索引到代码中

    def get_dir_Tx(self):
        self.dir_Tx = self.simuSetDict["dir_dataset"] + "png/antennas/"

    def get_dir_cars(self):
        self.dir_cars = self.simuSetDict["dir_dataset"] + "png/cars/"


    def get_image_gain(self,index):
        # 计算信源索引
        idxr=np.floor(index/self.simuSetDict["numTx"]).astype(int)               # 地图索引
        idxc=index-idxr*self.simuSetDict["numTx"]                                # 发射器索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1        # 实际地图编号（+1因为文件从1开始）
        source_name = str(dataset_map_ind) + "_" + str(idxc) + ".png"# 发射器相关文件

        # 如果使用rand模式则会进行两种方法的融合
        if self.simuSetDict["simulation"] != "rand":
            img_name_gain = os.path.join(self.dir_gain, source_name)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 255
        else:

            # 随机混合DPM和IRT2 对两种增益进行随机权重加权 IRT2的最大权重可设置为IRT2maxW
            img_name_gainDPM = os.path.join(self.dir_gainDPM, source_name)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, source_name)
            w = np.random.uniform(0, self.simuSetDict["IRT2maxW"])  # IRT2 weight of random average # 随机权重
            image_gain = w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256 \
                         + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256

        # pathloss threshold transform
        # 路径损耗阈值处理
        if self.simuSetDict["thresh"] > 0:
            mask = image_gain < self.simuSetDict["thresh"]
            image_gain[mask] = self.simuSetDict["thresh"]
            image_gain = image_gain - self.simuSetDict["thresh"] * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.simuSetDict["thresh"])

        image_gain = image_gain * 256

        return image_gain

    def get_image_buildings(self,index):
        # 计算地图索引
        idxr=np.floor(index/self.simuSetDict["numTx"]).astype(int)               # 地图索引                           # 发射器索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1        # 实际地图编号（+1因为文件从1开始）
        map_name = str(dataset_map_ind) + ".png"
        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, map_name)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings + str(self.missing) + "/" + str(version) + "/", map_name)

        image_buildings = np.asarray(io.imread(img_name_buildings))  # Will be normalized later, after random seed is computed from it
        image_buildings = image_buildings / 256

        return image_buildings

    def get_image_Tx(self,index):
        # 计算信源索引
        idxr=np.floor(index/self.simuSetDict["numTx"]).astype(int)               # 地图索引
        idxc=index-idxr*self.simuSetDict["numTx"]                                # 发射器索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1        # 实际地图编号（+1因为文件从1开始）
        source_name = str(dataset_map_ind) + "_" + str(idxc) + ".png"# 发射器相关文件
        img_name_Tx = os.path.join(self.dir_Tx, source_name)
        image_Tx = np.asarray(io.imread(img_name_Tx)) / 256

        return image_Tx

    def get_input_samples(self,image_gain):

        # input samples from the sparse gain samples
        image_samples = np.zeros((256,256)) # 创建全零采样图
        # 确定采样点数
        if self.simuSetDict["fix_samples"]==0:# 随机采样点数
            num_samples=np.random.randint(self.simuSetDict["num_samples_low"], self.simuSetDict["num_samples_high"], size=1)
        else:# 固定采样点数
            num_samples=np.floor(self.simuSetDict["fix_samples"]).astype(int)

        x_samples=np.random.randint(0, 255, size=num_samples)
        y_samples=np.random.randint(0, 255, size=num_samples)
        image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]

        return image_samples

    def get_cars_map(self,index):
        # 计算地图索引
        idxr=np.floor(index/self.simuSetDict["numTx"]).astype(int)               # 地图索引                           # 发射器索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1        # 实际地图编号（+1因为文件从1开始）
        map_name = str(dataset_map_ind) + ".png"
        img_name_cars = os.path.join(self.dir_cars, map_name)
        image_cars = np.asarray(io.imread(img_name_cars)) / 256

        return image_cars




    def __len__(self):
        return (self.ind2-self.ind1+1)*self.simuSetDict["numTx"]

    def __getitem__(self, idx):
        # 计算地图索引和发射器索引
        image_gain = self.get_image_gain(self, idx)
        input_samples = self.get_input_samples(image_gain)
        image_Tx = self.get_image_Tx(idx)
        image_buildings = self.get_image_buildings(idx)
        if self.simuSetDict["carsInput"] =="no":
            inputs = np.stack([image_buildings, image_Tx, input_samples], axis=2)
        else:
            image_cars = self.get_cars_map(idx)
            inputs = np.stack([image_buildings, image_Tx, input_samples, image_cars], axis=2)
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)

        return [inputs, image_gain]

    # self.numTx = simuSetDict['numTx']#信源数量
    # self.thresh = simuSetDict['thresh']# 环境噪声阈值
    # self.simulation = simuSetDict['simulation']# 仿真方式
    # self.carsSimul = simuSetDict['carsSimul']# 是否包含车辆
    # self.carsInput = simuSetDict['carsInput']# 输出是否使用 车辆位置作为模型输入
    # self.cityMap=simuSetDict["cityMap"]
    # self.missing=simuSetDict["missing"]
    #
    # self.data_samples = simuSetDict["data_samples"]
    # self.fix_samples = fix_samples
    # self.num_samples_low = num_samples_low
    # self.num_samples_high = num_samples_high
    
    