import torch
import torch.nn as nn
import torch.optim as optim


class pre_pi_GRU(nn.Module):
    def __init__(self,input_len ,input_dim,out_dim ):
        super(pre_pi_GRU, self).__init__()
        hidden_dim = 128
        # GRU层，hidden_dim为GRU单元的输出维度
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,num_layers=4, batch_first=True,bidirectional=True)

        # 输出层，返回的维度为input_dim
        self.fc1 = nn.Linear(hidden_dim*2*input_len, out_dim*10)
        self.fc2 = nn.Linear(out_dim*10, out_dim)

        self.gelu = nn.GELU()

    def forward(self, x):
        # GRU层的输出 (output, hidden_state)
        # 由于 return_sequences=False，所以我们只需要最后一个输出
        gru_out, _ = self.gru(x)

        gru_out = gru_out.reshape(gru_out.size(0), -1)

        # 通过全连接层输出，形状为 (batch_size, input_dim)
        output =  self.fc2(self.gelu(self.fc1(gru_out)))

        return output


def pre_pi_GRU_test( ):
    # 随机生成输入数据，形状为 (batch_size, input_len, input_dim)
    input_len = 1000
    input_dim = 10
    out_dim = 10
    batch_size = 64
    model =pre_pi_GRU(input_len ,input_dim,out_dim)
    test_input = torch.randn(batch_size, input_len, input_dim)

    # 将模型设置为评估模式
    model.eval()

    # 前向传播，获取输出
    with torch.no_grad():  # 禁用梯度计算，提高推理速度
        output = model(test_input)

    # 输出形状应该是 (batch_size, input_dim)
    print(f'Output shape: {output.shape}')

    return output


if __name__ == '__main__':
    pre_pi_GRU_test()
