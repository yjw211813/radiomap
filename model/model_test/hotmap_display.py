
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Grad-CAM 伪代码示例


def generate_grad_cam(model, img, target_class):
    # 前向传播
    logits = model(img)

    # 获取目标层（通常是最后一个卷积层）
    target_layer = model.layer4[-1].conv3

    # 计算梯度
    model.zero_grad()
    logits[0, target_class].backward()

    # 获取特征图和梯度
    features = target_layer.activations
    gradients = target_layer.gradients

    # 计算权重
    weights = torch.mean(gradients, dim=(2, 3))

    # 生成热力图
    cam = torch.sum(weights[:, :, None, None] * features, dim=1)
    cam = F.relu(cam)  # 移除负值
    cam = cam - cam.min()
    cam = cam / cam.max()  # 归一化

    return cam

# 以SE模块为例
class SEBlock(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        # 获取通道注意力权重
        weights = self.fc(self.pool(x))  # [b, c]
        # 可视化权重
        heatmap = weights.view(b, c, 1, 1).repeat(1, 1, h, w)
        return heatmap


# 加载预训练模型
model = YourModel(pretrained=True)
model.eval()

# 加载图像
img = cv2.imread("image.png")
img = preprocess(img)  # 标准化/缩放
features = {}
def hook_fn(module, input, output):
    features['target_layer'] = output.detach()

# 注册钩子（针对不同模块）
model.layer4[-1].conv3.register_forward_hook(hook_fn)  # 对于ResNet
# 或针对注意力模块
model.attn_block.register_forward_hook(hook_fn)

features = {}
def hook_fn(module, input, output):
    features['target_layer'] = output.detach()

# 注册钩子（针对不同模块）
model.layer4[-1].conv3.register_forward_hook(hook_fn)  # 对于ResNet
# 或针对注意力模块
model.attn_block.register_forward_hook(hook_fn)

# 前向传播
output = model(img)
target_class = output.argmax().item()

# 反向传播计算梯度
model.zero_grad()
output[0, target_class].backward()

# 获取梯度
gradients = torch.autograd.grad(outputs=output[0, target_class],
                               inputs=features['target_layer'])[0]
weights = torch.mean(gradients, dim=(2, 3))

# 计算CAM
cam = torch.sum(weights[:, :, None, None] * features['target_layer'], dim=1)
cam = torch.relu(cam)
cam = (cam - cam.min()) / (cam.max() - cam.min())

# 转换为numpy
cam = cam.squeeze().cpu().numpy()
cam = cv2.resize(cam, (img.shape[2], img.shape[1]))

# 创建热力图
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# 叠加到原图
result = heatmap * 0.5 + original_img * 0.5

# 显示
plt.imshow(result)
plt.axis('off')
plt.show()




