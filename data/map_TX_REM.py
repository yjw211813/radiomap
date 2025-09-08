from data.lib import REM_GAN_loaders
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from data.lib.seer_loader import RadioMapSeerLoader

def display_images(image_build_ant, image_gain,image_samples= None):

    # 根据 image_build_ant 的长度设置并排显示的图像数
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 默认2行3列
    axes = axes.ravel()  # 展平 axes 数组，方便索引
    for i in range(len(image_build_ant)):
        axes[i].imshow(image_build_ant[i], cmap='jet')
    i = i + 1
    # 显示掩码区域
    axes[i].imshow(image_gain[0], cmap='jet')

    # i = i + 1
    # # 显示掩码区域
    # axes[i].imshow(image_samples[0], cmap='jet')

    # 自动调整子图的布局
    plt.tight_layout()
    plt.show()

# 官方库
Radio_train = REM_GAN_loaders.RadioUNet_s(phase="train")
Radio_val = REM_GAN_loaders.RadioUNet_s(phase="val")
Radio_test = REM_GAN_loaders.RadioUNet_s(phase="test")

image_datasets = {
    'train': Radio_train, 'val': Radio_val
}

batch_size = 15


i=450
image_build_ant, image_gain = Radio_train[i]
display_images(image_build_ant, image_gain)
# 我写的库
#
# simuSetDict = {
#     "ind1": 0,  # 起始索引
#     "ind2": 0,  # 末尾索引
#     "dir_dataset": r"/home/data/path_loss_data/RadioSeer/RadioMapSeer/",  # 数据集文件夹
#     "numTx": 2,  # 信源数量设定
#     "thresh": 0.2,  # 环境噪声
#     "simulation": "IRT4",  # 模拟类型："DPM", "IRT2", "rand",如果是"IRT4" numTx必须小于2，如果大于 2 则强制设定为 2
#     "carsSimul": "no",  # 是否开启小车作为仿真
#     "carsInput": "no",  # 是否将小车图作为模型输入
#     "IRT2maxW": 1,  # 如果simulation是rand 表明是融合DPM和IRT2 IRT2maxW这为最大的加权值
#     "cityMap": "complete",  # 是否输入完全的城市地图
#     "missing": 1,  # 地图缺失号码
#     "fix_samples": 300,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
#     "num_samples_low": 10,  # 最低采样数
#     "num_samples_high": 300,  # 最高采样数
#     "inter_flag": False,  # 看是否需要插值图像
#     "sample_flag": False,
#     "scale256_flag": False,  # 取值范围是否为0 - 255
#     "loss_samples_flag":True,# 是否定义loss为稀疏采样loss
# }
# # 加载数据集
# Radio_train = RadioMapSeerLoader(simuSetDict, phase="train")
#
# inputs, image_gain,image_samples = Radio_train[i]
#
# display_images(inputs, image_gain,image_samples)
