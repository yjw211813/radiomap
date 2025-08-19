import torch
import torch.nn as nn


def test_softmax_dim():
    # 1. 构造测试数据 (batch=2, num_branches=3, C_out=4)
    test_input = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0],  # 分支1的4个通道
         [0.5, 1.5, 2.5, 3.5],  # 分支2
         [1.5, 0.5, 2.0, 3.0]],  # 分支3

        [[-1.0, 0.0, 1.0, 2.0],
         [0.0, -0.5, 0.5, 1.5],
         [1.0, 1.0, 1.0, 1.0]]
    ], dtype=torch.float32)

    print("输入张量形状:", test_input.shape)  # torch.Size([2, 3, 4])

    # 2. 初始化Softmax层
    softmax = nn.Softmax(dim=-1)  # 对最后一个维度(C_out)计算

    # 3. 前向计算
    output = softmax(test_input)

    # 4. 验证输出形状
    print("输出张量形状:", output.shape)  # 应保持 torch.Size([2, 3, 4])

    # 5. 验证最后一个维度求和为1
    sum_last_dim = output.sum(dim=-1)
    print("最后一个维度求和:\n", sum_last_dim)
    print("是否全为1:", torch.allclose(sum_last_dim, torch.ones_like(sum_last_dim)))

    # 6. 手动计算第一个样本的第一个分支的softmax
    first_branch = test_input[0, 0]  # tensor([1., 2., 3., 4.])
    manual_softmax = torch.exp(first_branch) / torch.exp(first_branch).sum()
    print("PyTorch计算结果:", output[0, 0])
    print("手动计算结果:", manual_softmax)
    print("是否一致:", torch.allclose(output[0, 0], manual_softmax))

    # 7. 验证其他维度是否独立计算（比较分支间的值）
    print("\n验证分支间独立性:")
    print("分支1 softmax:", output[0, 0])
    print("分支2 softmax:", output[0, 1])
    print("注意：不同分支的softmax结果互不影响")


if __name__ == "__main__":
    test_softmax_dim()