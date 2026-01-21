from data.lib import loaders
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from data.lib.seer_loader import RadioMapSeerLoader
'''

| flag组合（sample_flag / formula_flag / inter_flag / carsInput） | inputs 包含的内容（按顺序，简称对应原始数据） |
|------------------------------------|------------------------------------------------|
| False / - / - / "no"               | 【building（image_buildings）, source_tx（image_Tx）】 |
| False / - / - / ≠"no"              | 【building, source_tx, car_img（image_cars）】 |
| True / False / False / "no"        | 【building, source_tx, sample（input_samples）】 |
| True / False / False / ≠"no"       | 【building, source_tx, sample, car_img】 |
| True / True / False / "no"         | 【building, source_tx, sample, fomula_img（genImg）】 |
| True / True / False / ≠"no"        | 【building, source_tx, sample, fomula_img, car_img】 |
| True / False / True / "no"          | 【building, source_tx, sample, inter_img（interpolate_data）】 |
| True / False / True / ≠"no"         | 【building, source_tx, sample, inter_img, car_img】 |
| True / True / True / "no"           | 【building, source_tx, sample, fomula_img, inter_img】 |
| True / True / True / ≠"no"          | 【building, source_tx, sample, fomula_img, inter_img, car_img】（示例对应场景） |


'''


def display_images(image_build_ant, image_gain):
    # 获取 image_build_ant 的长度
    build_ant_len = len(image_build_ant)

    # 根据 image_build_ant 的长度设置并排显示的图像数
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # 默认2行3列
    axes = axes.ravel()  # 展平 axes 数组，方便索引
    for i in range(len(image_build_ant)):
        axes[i].imshow(image_build_ant[i], cmap='jet')
    i = i + 1
    # 显示掩码区域
    axes[i].imshow(image_gain[0], cmap='jet')


    # 自动调整子图的布局
    plt.tight_layout()
    plt.show()

# # 官方库
# Radio_train = loaders.RadioUNet_c_sprseIRT4(phase="train")
# Radio_val = loaders.RadioUNet_c_sprseIRT4(phase="val")
# Radio_test = loaders.RadioUNet_c_sprseIRT4(phase="test")
#
# image_datasets = {
#     'train': Radio_train, 'val': Radio_val
# }
#
# batch_size = 15



i=454
# image_build_ant, image_gain,image_samples = Radio_train[i]
# display_images(image_build_ant, image_gain,image_samples)
# 我写的库

sample_rate_max = 0.03
sample_rate_min = 0.005
num_samples_high = int(256*256*sample_rate_max)
num_samples_low = int(256*256*sample_rate_min)

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
    "fix_samples": 0,  # 采样数量 如果为0 则随机一个采样数 下面是随机范围 如果不为0则使用固定的采样数
    "num_samples_low": num_samples_low,  # 最低采样数
    "num_samples_high": num_samples_high,  # 最高采样数
    "inter_flag": False,  # 看是否需要插值图像
    "scale256_flag": True,  # 取值范围是否为0 - 255
    "sample_flag": True,  # 是否有采样输入
    "loss_samples_flag": False,  # 是否定义loss为稀疏采样loss
    "formula_flag": False
}
# 加载数据集
Radio_train = RadioMapSeerLoader(simuSetDict, phase="train")

inputs, image_gain = Radio_train[i]

display_images(inputs, image_gain)
