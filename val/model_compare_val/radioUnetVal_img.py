import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.radioUnetModel import modules  # 导入自定义模型模块
from data.lib import loaders  # 导入数据加载器

# =============== 配置参数 ===============
model_save_dir = "/home/code/radio_map_construction/runs/model_compare/radioUnet/"
img_save_dir = "/home/code/radio_map_construction/runs/model_compare/radioUnet/example/"
batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============== 创建输出目录 ===============
os.makedirs(img_save_dir, exist_ok=True)


# =============== 图像生成函数 ===============
def create_visualization(data_array, building_mask, title_suffix, save_path):
    """创建三通道可视化图像并保存"""
    vis = np.zeros((256, 256, 3), dtype=np.uint8)
    vis[..., 0] = data_array  # 红色通道
    vis[..., 1] = data_array  # 绿色通道
    vis[..., 2] = data_array  # 蓝色通道
    vis[building_mask, 2] = 100  # 建筑物区域标记为蓝色

    img = Image.fromarray(vis)
    img.save(save_path)
    return save_path


# =============== 主程序 ===============
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 1. 加载测试数据集
    Radio_test = loaders.RadioUNet_c(phase="test")

    # 3. 初始化模型（选择第二阶段U-Net）
    model = modules.RadioWNet(phase="secondU")
    state_dict = torch.load(os.path.join(model_save_dir, "Trained_Model_SecondU.pt"), map_location=device)
    model.load_state_dict(state_dict)
    print("pt load sucess")
    model.to(device)  # 将模型移至计算设备
    model.eval()  # 设置为评估模式


    # 2. 准备测试索引
    maps_inds = np.arange(700, dtype=np.int16)
    np.random.seed(42)
    np.random.shuffle(maps_inds)

    # 3. 处理测试样本
    with torch.no_grad():
        for map_idx in range(1, 99):
            # 创建数据集和加载器
            data_idx = maps_inds[600 + map_idx]
            dataset = loaders.RadioUNet_c(maps_inds=maps_inds, phase="custom",
                                          ind1=600 + map_idx, ind2=600 + map_idx)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # 处理每个样本
            for sample_idx, (inputs, target) in enumerate(loader):
                if sample_idx >= 2:  # 每张地图只处理两个样本
                    break

                inputs = inputs.to(device)
                target = target.to(device)

                # 模型预测
                _, pred = model(inputs)
                pred_np = (255 * pred.squeeze().cpu().numpy()).astype(np.uint8)
                target_np = (255 * target.squeeze().cpu().numpy()).astype(np.uint8)
                building_mask = (inputs.squeeze().cpu().numpy()[0] != 0)

                # 生成文件名
                base_name = f"{map_idx}_{data_idx}_{sample_idx + 1}"
                # 保存预测结果
                pred_path = os.path.join(img_save_dir, f"{base_name}_DPM_predict.png")
                create_visualization(pred_np, building_mask, "Predicted", pred_path)
                # 保存真实结果
                target_path = os.path.join(img_save_dir, f"{base_name}_DPM_target.png")
                create_visualization(target_np, building_mask, "Target", target_path)

    # =============== 创建对比图 ===============
    # 配置要展示的样本
    samples = [
        {"id": "17_187_1", "title": "Sample 1 (Map 17)"},
        {"id": "41_508_2", "title": "Sample 2 (Map 41)"},
        {"id": "31_476_2", "title": "Sample 3 (Map 31)"}
    ]

    # 创建对比图表
    fig, axs = plt.subplots(3, 4, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for row, sample in enumerate(samples):
        sample_id = sample["id"]

        # 加载四种图像类型
        image_types = {
            "DPM_target": f"{sample_id}_DPM_target.png",
            "DPM_predict": f"{sample_id}_DPM_predict.png",
        }

        # 显示当前样本的四种图像
        for col, (img_type, img_name) in enumerate(image_types.items()):
            img_path = os.path.join(img_save_dir, img_name)
            img = plt.imread(img_path)

            ax = axs[row, col]
            ax.imshow(img)
            ax.axis('off')

            # 设置列标题（第一行）
            if row == 0:
                ax.set_title(img_type.replace('_', ' ').title())

            # 设置行标题（第一列）
            if col == 0:
                ax.set_ylabel(sample["title"], rotation=90, size='large')

    # 保存和显示图表
    plt.savefig(os.path.join(img_save_dir, "comparison_summary.png"))
    plt.tight_layout()
    plt.show()