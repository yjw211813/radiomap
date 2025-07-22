from pydoc import importfile
import sys
import os
import h5py
import numpy as np
import torch
from openpyxl.styles.builtins import output
from tensorboard.compat.tensorflow_stub.dtypes import float32
from torch.utils.data import Dataset, DataLoader
# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.sub_block.conv2D_block import *
from concurrent.futures import ThreadPoolExecutor
from environment_code.tif_convert_height import map_info
from environment_code.generate_radio_map_data import data_structure,simu_visual
import matplotlib.pyplot as plt
from data.format_image import RadioMap
import time

# 计算两点间的距离
def calculate_distance(pos1, pos2):

    # 地球半径（米）
    R = 6371000
    lon1,lat1,alt1 = pos1
    lon2,lat2,alt2 = pos2
    # 将角度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    # Haversine公式计算球面距离
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    # 球面距离
    surface_distance = R * c
    # 高度差
    height_diff = alt2 - alt1
    # 三维空间距离（使用勾股定理）
    distance = math.sqrt(surface_distance ** 2 + height_diff ** 2)
    return distance

# 测试距离计算程序
def test_calculate_distance():
    # 例如：北京（39.9042° N, 116.4074° E, 43m）到上海（31.2304° N, 121.4737° E, 4m）
    pos1 = 116.4074, 39.9042 , 43  # 北京
    pos2= 121.4737,31.2304   , 4  # 上海

    distance = calculate_distance(pos1, pos2)
    print(f"两点之间的距离为: {distance:.2f} 米")
# 利用hata模型计算距离
def Hata_path_loss(city_type, f, hr, ht, d):
    """
    计算路径损耗 PL
    city_type (str): 城市类型 ('small', 'large', 'suburban', 'rural')
    f (float): 频率 (MHz)
    hr (float): 接收天线高度 (米)
    ht (float): 发射天线高度 (米)
    d (float): 距离 (公里)
    distance_loss (float): 路径损耗 (dB)
    """
    # 计算 C 和 a_hr
    if city_type == 'small':  # 中小城市
        C = -4.78 * (math.log10(f))**2 + 18.33 * math.log10(f) - 40.98
        ahr = (1.1 * math.log10(f) - 0.7) * hr - (1.56 * math.log10(f) - 0.8)
    elif city_type == 'large':  # 大城市
        C = 0
        if f <= 200:
            ahr = 8.29 * (math.log10(1.54 * hr))**2 - 1.1
        else:
            ahr = 3.2 * (math.log10(11.75 * hr))**2 - 4.97
    elif city_type == 'suburban':  # 郊区
        C = -2 * (math.log10(f / 28))**2 - 5.4
        ahr = (1.1 * math.log10(f) - 0.7) * hr - (1.56 * math.log10(f) - 0.8)
    elif city_type == 'rural':  # 农村
        C = 4.78 * (math.log10(f))**2 + 18.33 * math.log10(f) - 40.98
        ahr = (1.1 * math.log10(f) - 0.7) * hr - (1.56 * math.log10(f) - 0.8)
    else:
        raise ValueError("Invalid city type. Choose from: 'small', 'large', 'suburban', 'rural'.")

    # 计算 A 和 B
    A = 69.55 + 26.16 * math.log10(f) - 13.82 * math.log10(ht) - ahr
    B = 44.9 - 6.55 * math.log10(ht)

    # 计算路径损耗 PL
    distance_loss = A + B * math.log10(d) + C + 10040

    return distance_loss

# 数模双驱 输入为信源特征 随机采样点
class formula_map(nn.Module):
    def __init__(self,feature_num,img_H,img_W):
        super(formula_map, self).__init__()
        # 输入480*480
        # 输入参数定义
        C_out1 = 32
        down_num = 5
        self.hidden_mul1 = 10
        self.hidden_mul2 = 20
        self.hidden_mul_bias = 20
        self.feature_num = feature_num
        self.img_H = img_H
        self.img_W = img_W

        self.encoder = nn.Sequential(
            Inception_ghost2D(C_in=1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),   #  下采样 (100,C_out1,240,240)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),   #  下采样 (100,C_out1,120,120)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),  #  下采样 (100,C_out1,60,60)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),  #  下采样 (100,C_out1,30,30)
            Fractal_inception2D(input_channel=C_out1, output_channel=C_out1),
            Conv_DownSampling2D(C_out1)  #  下采样 (100,C_out1,15,15)
        )
        out_H = int(img_H / (2 ** down_num))
        out_W = int(img_W / (2 ** down_num))

        self.fc_weight = nn.Linear(C_out1 * out_H * out_W, feature_num*self.hidden_mul1 * feature_num*self.hidden_mul2)

        self.fc1 = nn.Linear(feature_num, feature_num*self.hidden_mul1)
        self.fc2 = nn.Linear(feature_num*self.hidden_mul1, feature_num*self.hidden_mul2)
        self.fc_map =nn.Linear(feature_num*self.hidden_mul2, img_H*img_W)

        self.fc_bias1 = nn.Linear(feature_num, feature_num*self.hidden_mul_bias)
        self.fc_bias2 = nn.Linear(feature_num*self.hidden_mul_bias, 1)
        self.gelu = nn.GELU()

    def forward(self, samp_map,Hata_map,feature_in):
        feature_in = feature_in.squeeze(1)
        batch_size = samp_map.shape[0]
        out_weight = self.fc_weight(self.encoder(samp_map).reshape(batch_size,-1))
        out_weight = out_weight.reshape(batch_size, self.feature_num*self.hidden_mul1, self.feature_num * self.hidden_mul2)
        feature_out = self.gelu(self.fc1(feature_in))

        feature_out = torch.matmul(feature_out, out_weight).squeeze(1)
        out_map = self.gelu(self.fc_map(feature_out))
        bias = self.fc_bias2(self.gelu(self.fc_bias1(feature_in))).unsqueeze(1)
        out_map =  out_map.reshape(batch_size,1,Hata_map.shape[2],Hata_map.shape[3])
        out_map = out_map + Hata_map + bias

        return out_map
# 模型测试程序
def formula_map_forward_test():
    feature_num = 4
    img_H = 480
    img_W = 480
    feature_data = torch.rand(5,1,1,feature_num)
    img_data = torch.rand(5, 1, img_H, img_W)
    Hata_map = torch.rand(5, 1, img_H, img_W)
    net = formula_map(feature_num,img_H,img_W)
    torch.save(net.state_dict() , f"../model_path/checkpoint_epoch.pth")
    output = net(img_data,Hata_map,feature_data)
    print("output shape :",output.shape)

def formula_map_test():
    base_path = '../environment_code/'
    json_filename = base_path + "simu_data_info.json"
    tif_path = base_path + "output_dem.tif"
    map_pkl_path = base_path + "geo_map.pkl"
    dir_name = r'../environment_code/surface_simu'
    config_json_path = base_path + "map_config.json"
    feature_num = 4
    set_H = 480
    set_W = 480
    data_info = data_structure(div_pos=25,
                               div_angle=12,
                               json_filename=json_filename,
                               dir_name=dir_name)

    map_object = map_info(tif_file=tif_path,
                          map_file=map_pkl_path,
                          config_json=config_json_path)
    map_object.get_map_data()
    map_object.reshape(set_H, set_W)
    sample_ratio = 0.01

    rand_img, Hata_PL, site_info, ori_img = process_sample_test(data_info, map_object, sample_ratio, set_H, set_W)
    plot_map('随机采样点数据', rand_img, site_info)
    plot_map('Hata 路劲损失计算', Hata_PL, site_info)
    plot_map('原始数据', ori_img, site_info)

    net = formula_map(feature_num, set_H, set_W)
    rand_img = rand_img.unsqueeze(0).unsqueeze(0).to(torch.float32)
    Hata_PL = Hata_PL.unsqueeze(0).unsqueeze(0).to(torch.float32)
    site_info = torch.from_numpy(site_info).to(torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    output = net(rand_img, Hata_PL, site_info)

    print("output shape :", output.shape)

# 三维乘法广播测试
def matrix_mul_test():
    # 定义张量 A 和 B
    A = torch.randn(5, 3, 4)  # (5, 3, 4)
    B = torch.randn(5, 4, 6)  # (5, 4, 6)
    # 初始化一个空的张量来保存结果
    C_manual = torch.zeros(5, 3, 6)  # (5, 3, 6)
    # 使用 for 循环逐个批次计算矩阵乘法
    for i in range(5):
        C_manual[i,:,:] = torch.matmul(A[i,:,:], B[i,:,:])
    # 输出手动计算的结果形状
    print("Manual batch matrix multiplication result shape:", C_manual.shape)
    # 验证与 torch.matmul 的结果是否相同
    C_torch = torch.matmul(A, B)
    print("torch.matmul batch matrix multiplication result shape:", C_torch.shape)
    # 验证两者是否相同
    print("Are both results equal? ", torch.allclose(C_manual, C_torch))
# 数模双驱 输入为信源特征 随机采样点 还有卫星地图
class formula_satelite_map(nn.Module):
    def __init__(self,feature_num,img_H,img_W):
        super(formula_satelite_map, self).__init__()
        # 输入480*480
        # 输入参数定义
        C_out1 = 32
        down_num = 5
        self.hidden_mul1 = 10
        self.hidden_mul2 = 20
        self.hidden_mul_bias = 20
        self.feature_num = feature_num
        self.img_H = img_H
        self.img_W = img_W

        self.encoder = nn.Sequential(
            Inception_ghost2D(C_in=2, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),   #  下采样 (100,C_out1,240,240)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),   #  下采样 (100,C_out1,120,120)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),  #  下采样 (100,C_out1,60,60)
            Inception_ghost2D(C_in=C_out1, C_out=C_out1, kernel_sizes=[1, 3, 5, 7], dilated_num=1),
            Conv_DownSampling2D(C_out1),  #  下采样 (100,C_out1,30,30)
            Fractal_inception2D(input_channel=C_out1, output_channel=C_out1),
            Conv_DownSampling2D(C_out1)  #  下采样 (100,C_out1,15,15)
        )
        out_H = int(img_H / (2 ** down_num))
        out_W = int(img_W / (2 ** down_num))

        self.fc_weight = nn.Linear(C_out1 * out_H * out_W, feature_num*self.hidden_mul1 * feature_num*self.hidden_mul2)

        self.fc1 = nn.Linear(feature_num, feature_num*self.hidden_mul1)
        self.fc2 = nn.Linear(feature_num*self.hidden_mul1, feature_num*self.hidden_mul2)
        self.fc_map =nn.Linear(feature_num*self.hidden_mul2, img_H*img_W)

        self.fc_bias1 = nn.Linear(feature_num, feature_num*self.hidden_mul_bias)
        self.fc_bias2 = nn.Linear(feature_num*self.hidden_mul_bias, 1)
        self.gelu = nn.GELU()


    def forward(self, Input_img,Hata_map,feature_in):

        feature_in = feature_in.squeeze(1)
        batch_size = Input_img.shape[0]
        out_weight = self.fc_weight(self.encoder(Input_img).reshape(batch_size,-1))
        out_weight = out_weight.reshape(batch_size, self.feature_num*self.hidden_mul1, self.feature_num * self.hidden_mul2)
        feature_out = self.gelu(self.fc1(feature_in))

        feature_out = torch.matmul(feature_out, out_weight).squeeze(1)
        out_map = self.gelu(self.fc_map(feature_out))
        bias = self.fc_bias2(self.gelu(self.fc_bias1(feature_in))).unsqueeze(1)
        out_map =  out_map.reshape(batch_size,1,Hata_map.shape[2],Hata_map.shape[3])
        out_map = out_map + Hata_map + bias

        return out_map

def formula_satelite_forward_test():
    feature_num = 4
    img_H = 480
    img_W = 480
    feature_data = torch.rand(5,1,1,feature_num)
    img_data = torch.rand(5, 2, img_H, img_W)
    Hata_map = torch.rand(5, 1, img_H, img_W)
    net = formula_satelite_map(feature_num,img_H,img_W)
    torch.save(net.state_dict(), f"../model_pth/formula_satelite_map_train/checkpoint_epoch.pth")
    output = net(img_data,Hata_map,feature_data)
    print("output shape :",output.shape)
def formula_satelite_map_test():

    base_path = '../../data/map_data/'
    json_filename = base_path + "simu_data_info.json"
    tif_path = base_path + "output_dem.tif"
    map_pkl_path = base_path + "geo_map.pkl"
    dir_name = r'../../data/path_loss_data/surface_simu'
    config_json_path = base_path + "map_config.json"
    feature_num = 4
    set_H = 480
    set_W = 480
    data_info = data_structure(div_pos=25,
                               div_angle=12,
                               json_filename=json_filename,
                               dir_name=dir_name)

    map_object = map_info(tif_path=tif_path,
                          map_pkl_path=map_pkl_path,
                          config_json_path=config_json_path)
    map_object.get_map_data()
    map_object.reshape(set_H, set_W)
    sample_ratio = 0.1

    rand_img, Hata_PL, site_info, ori_img = process_sample_test(data_info, map_object, sample_ratio, set_H, set_W)
    map_data = map_object.geo_map
    plot_map('随机采样点数据', rand_img, site_info)
    plot_map('Hata 路劲损失计算', Hata_PL, site_info)
    plot_map('原始数据', ori_img, site_info)
    plot_map('地图数据', map_data, site_info)
    net = formula_satelite_map(feature_num, set_H, set_W)
    map_data = map_data.unsqueeze(0).unsqueeze(0).to(torch.float32)
    rand_img = rand_img.unsqueeze(0).unsqueeze(0).to(torch.float32)
    combined_data = torch.cat((map_data, rand_img), dim=1)
    Hata_PL = Hata_PL.unsqueeze(0).unsqueeze(0).to(torch.float32)
    site_info = torch.from_numpy(site_info).to(torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    output = net(combined_data, Hata_PL, site_info)

    print("output shape :", output.shape)

def formula_satelite_map_torchscript_save():
    feature_num = 4
    img_H = 480
    img_W = 480
    feature_data = torch.rand(5, 1, 1, feature_num)
    img_data = torch.rand(5, 2, img_H, img_W)
    Hata_map = torch.rand(5, 1, img_H, img_W)
    net = formula_satelite_map(feature_num, img_H, img_W)
    traced_script_module = torch.jit.trace(net, (img_data,Hata_map,feature_data))
    traced_script_module.save("formula_satelite_map_torchscript.pt")
    output = net(img_data, Hata_map, feature_data)
    print("output shape :", output.shape)

def formula_satelite_map_torchscript_load():
    feature_num = 4
    img_H = 480
    img_W = 480
    feature_data = torch.rand(5, 1, 1, feature_num)
    img_data = torch.rand(5, 2, img_H, img_W)
    Hata_map = torch.rand(5, 1, img_H, img_W)

    loaded_script_module = torch.jit.load("formula_satelite_map_torchscript.pt")
    output = loaded_script_module(img_data, Hata_map, feature_data)

    print("output shape :", output.shape)

# 利用Hata 模型计算路径损失
def get_Hata_map(map_object,source):
    s_height = map_object.get_h_lonlat(source.lon, source.lat)
    rec_height = 4500
    set_H = 480
    set_W = 480
    path_loss_matrix = np.zeros_like(map_object.geo_map)
    for row in range(path_loss_matrix.shape[0]):
        for col in range(path_loss_matrix.shape[1]):
            rec_lon = map_object.lon_min + row * map_object.lon_len
            rec_lat = map_object.lat_min + col * map_object.lat_len
            d = calculate_distance([source.lon, source.lat, s_height], [rec_lon, rec_lat, rec_height])
            path_loss_matrix[row, col] = Hata_path_loss(city_type="rural", f=np.exp(source.log_fre)
                                                    , hr=rec_height, ht=s_height, d=d) + 3480
    power_temp = RadioMap(path_loss_matrix, 0, 0, 0, 0)
    power_temp = power_temp.reshape(set_H,set_W)
    return power_temp.radio_data

# 处理单个样本数据
def process_sample(data_info, map_object, sample_ratio,set_H,set_W):
    """处理每个样本的获取与计算"""
    global progress_bar

    pathloss_map, site_info, source = data_info.get_one_path_loss(set_H,set_W)
    rand_img = pathloss_map.random_sample(sample_ratio).radio_data
    ori_img = pathloss_map.radio_data
    Hata_PL = get_Hata_map(map_object, source)
    progress_bar.update(1)  # 每次扩展后更新进度条
    return rand_img, Hata_PL, site_info, ori_img

def process_sample_test(data_info, map_object, sample_ratio,set_H,set_W):
    """处理每个样本的获取与计算"""
    global progress_bar

    pathloss_map, site_info, source = data_info.get_one_path_loss(set_H,set_W)
    rand_img = pathloss_map.random_sample(sample_ratio).radio_data
    ori_img = pathloss_map.radio_data
    Hata_PL = get_Hata_map(map_object, source)
    # progress_bar.update(1)  # 每次扩展后更新进度条
    return rand_img, Hata_PL, site_info, ori_img
# 获得随机路径损失数据进行模型训练
def get_randPL_map(data_info, map_object, sample_ratio,set_H,set_W, num_samples=100, num_threads=10):
    """使用多线程获取路径损耗图并保存"""
    rand_imgs = []
    hata_pls = []
    site_infos = []
    ori_imgs = []
    # 使用线程池来并行化样本获取过程
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 使用生成器来处理每个样本的并行任务
        futures = [executor.submit(process_sample, data_info, map_object, sample_ratio,set_H,set_W) for _ in range(num_samples)]
        for future in futures:
            # 获取每个任务的结果
            rand_img, Hata_PL, site_info, ori_img = future.result()
            # 将结果存入相应的列表
            rand_imgs.append(rand_img)
            hata_pls.append(Hata_PL)
            site_infos.append(site_info)
            ori_imgs.append(ori_img)
    # 将结果存入 HDF5 文件
    with h5py.File('../data/path_loss/train_radio_PL_Data.h5', 'w') as f:
        f.create_dataset('rand_imgs', data=np.array(rand_imgs))
        f.create_dataset('Hata_PLs', data=np.array(hata_pls))
        f.create_dataset('site_infos', data=np.array(site_infos))
        f.create_dataset('ori_imgs', data=np.array(ori_imgs))

    print(f'{num_samples} samples saved to train_radio_PL_Data.h5.')

# 打包训练测试数据
def pkl_data():
    json_filename = "../environment_code/simu_data_info.json"
    tif_file = '../environment_code/output_dem.tif'
    map_file = '../environment_code/geo_map.pkl'
    dir_name = r'..\environment_code\surface_simu'
    config_json = '../environment_code/map_config.json'
    from tqdm import tqdm

    feature_num = 4
    set_H = 480
    set_W = 480
    num_samples = 5
    # 创建进度条对象
    progress_bar = tqdm(total=num_samples, desc="打包进度", unit="项")
    data_info = data_structure(div_pos = 25,div_angle=12,
                               json_filename = json_filename,dir_name=dir_name)
    source = data_info.generate_one_pos()
    map_object = map_info(tif_file = tif_file,
                          map_file=map_file ,
                          config_json = config_json)
    map_object.get_map_data()
    map_object.reshape(set_H,set_W)
    sample_ratio = 0.1
    get_randPL_map(data_info = data_info,
                   map_object = map_object,
                   sample_ratio = sample_ratio,
                   set_H = set_H,
                   set_W = set_W,
                   num_samples=num_samples)
    rand_img, Hata_PL, site_info, ori_img = process_sample(data_info, map_object, sample_ratio,set_H,set_W)


def plot_map(description,map_data,site_info):
    visual_obj = simu_visual()
    visual_obj.set_style()
    visual_obj.plot_map(description,map_data)
    visual_obj.plot_source(site_info)
    visual_obj.show_object()

if __name__ == "__main__":
    formula_map_forward_test()

    print(1)
    # plot_map(map_data, site_info)

    # formula_map_forward_test()
    # formula_satelite_forward_test()

    # formula_satelite_map_test()
    # formula_satelite_map_torchscript_save()
    # formula_satelite_map_torchscript_load()


