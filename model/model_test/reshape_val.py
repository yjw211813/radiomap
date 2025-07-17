
import torch
import torch.nn.functional as F

def test_reshape():
    # 假设参数
    batch_size = 2
    img_H, img_W = 3, 4  # 目标形状：1x3x4
    d_model = 8  # 假设嵌入维度

    # 模拟 timembedding 的输出 (batch_size, img_H*img_W)
    emb = torch.randn(batch_size, img_H * img_W)
    print("Original shape:", emb.shape)  # 应为 (2, 12)

    # 方法1: 使用 reshape
    reshaped_emb = emb.reshape(-1, 1, img_H, img_W)
    print("Reshaped shape:", reshaped_emb.shape)  # 应为 (2, 1, 3, 4)

    # 验证数据是否一致
    print("\n验证reshape是否正确:")
    for i in range(batch_size):
        original_data = emb[i]  # 原始数据 (12,)
        reshaped_data = reshaped_emb[i, 0]  # 变形后数据 (3, 4)

        # 检查变形后的数据是否等于原始数据重组后的结果
        assert torch.allclose(
            original_data,
            reshaped_data.reshape(-1)  # 重新展平
        ), f"样本 {i} 数据不匹配！"
        print(f"样本 {i} 验证通过！")

    # 验证样本间是否独立（修改一个样本，另一个不应受影响）
    print("\n验证样本独立性:")
    sample0_original = reshaped_emb[0].clone()  # 保存样本0
    reshaped_emb[1] += 1.0  # 修改样本1
    assert torch.allclose(
        sample0_original,
        reshaped_emb[0]
    ), "样本0被样本1的修改影响了！"
    print("样本独立性验证通过！")




def test_channel_independence():
    # 创建输入张量：batch=10, channel=3, height=256, width=256
    input_tensor = torch.zeros(10, 3, 256, 256)

    # 测试1：只激活通道0
    input_tensor[:, 0, :, :] = 1.0  # 第0通道全1，其他通道保持0
    resized = F.interpolate(input_tensor, size=(32, 32), mode='bilinear', align_corners=True)

    # 验证：只有通道0应有非零值
    assert torch.all(resized[:, 1:, :, :].abs() < 1e-6), "通道0影响了其他通道！"
    assert resized[:, 0, :, :].min() > 0.99, "通道0数据异常"

    # 测试2：只激活通道1
    input_tensor.zero_()  # 重置为全0
    input_tensor[:, 1, :, :] = 1.0  # 第1通道全1
    resized = F.interpolate(input_tensor, size=(32, 32), mode='bilinear', align_corners=False)

    # 验证：只有通道1应有非零值
    assert torch.all(resized[:, 0, :, :].abs() < 1e-6), "通道1影响了通道0！"
    assert torch.all(resized[:, 2, :, :].abs() < 1e-6), "通道1影响了通道2！"
    assert resized[:, 1, :, :].min() > 0.99, "通道1数据异常"

    # 测试3：批量样本独立性 (检查batch维度)
    input_tensor.zero_()
    input_tensor[3, 2, :, :] = 1.0  # 仅第4个样本的第2通道置1
    resized = F.interpolate(input_tensor, size=(32, 32), mode='bilinear', align_corners=False)

    # 验证：只有batch=3, channel=2有值
    for b in range(10):
        for c in range(3):
            if b == 3 and c == 2:
                assert resized[b, c].min() > 0.98
            else:
                assert torch.all(resized[b, c].abs() < 1e-6)

    print("所有测试通过！通道变换完全独立。")


if __name__ == '__main__':
    test_channel_independence()