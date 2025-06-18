from model.conv1D_block import *

###  time features : (10,96,30)
class time_feature_embedding(nn.Module):
    def __init__(self,out_dim,input_H,input_W):
        super(time_feature_embedding, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model = input_W,
                                                      nhead = 5,
                                                      dim_feedforward=256,
                                                      dropout=0.1,
                                                      activation='gelu',
                                                      batch_first=True,
                                                      bias=True)
        self.time_embed =  nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc1 = nn.Linear(input_W*input_H,out_dim)

    def forward(self, x):
        out1 = self.time_embed(x)
        out1_flat = out1.reshape(out1.size(0), -1)
        out2 = self.fc1(out1_flat)
        return out2

###  time features : (10,96,30)
def time_feature_embedding_test():
    data = torch.randn(10,96,30)
    block = time_feature_embedding(out_dim = 30,input_H = 96,input_W = 30)
    output_tensor = block(data)
    # 打印输出形状
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output_tensor.shape}")

##### fre features : (100,8,1,100)
class fre_embedding(nn.Module):
    def __init__(self,out_dim,input_channels,length):
        super(fre_embedding, self).__init__()
        C_out1 = 64
        down_sample_times = 3
        ###  下采样 (100,C_out1,1,100)
        self.Fra_conv = Fractal_inception(input_channel=input_channels, output_channel=C_out1)
        self.down1 = Conv_DownSampling(C_out1) ###  下采样 (100,C_out1,1,50)
        self.conv_mixer1_1 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer1_2 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)

        self.down2 = Conv_DownSampling(C_out1)###  下采样 (100,C_out1,1,25)
        self.conv_mixer2_1 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer2_2 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.change_shape1 = nn.Conv2d(C_out1, C_out1, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0))
        ## (100, C_out1, 1, 24)

        self.down3 = Conv_DownSampling(C_out1)  ###  ## (100, C_out1, 1, 12)
        self.conv_mixer3_1 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer3_2 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)

        vector_len = int(length/(2**down_sample_times))
        encoder_layer = nn.TransformerEncoderLayer(d_model = vector_len,
                                                      nhead = 4,
                                                      dim_feedforward=128,
                                                      dropout=0.1,
                                                      activation='gelu',
                                                      batch_first=True,
                                                      bias=True)
        self.time_embed =  nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(C_out1*vector_len,out_dim)

    def forward(self, x):
        out1 = self.conv_mixer1_2(self.conv_mixer1_1(self.down1(self.Fra_conv(x) ) ) )
        out2 = self.change_shape1(self.conv_mixer2_2(self.conv_mixer2_1(self.down2(out1))))
        out3 = self.conv_mixer3_2(self.conv_mixer3_1(self.down3(out2)))
        out4 = self.time_embed(out3.squeeze(2))
        out4_flat = out4.reshape(out4.size(0), -1)
        out5 = self.fc1(out4_flat)
        return out5

###  time features : (100,8,1,100)
def fre_embedding_test():
    data = torch.randn(100,8,1,100)
    block = fre_embedding(out_dim = 20,input_channels = 8,length = 100)
    output_tensor = block(data)
    # 打印输出形状
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output_tensor.shape}")

##### wt2 features : (10,256,1,200)
class WT_embedding(nn.Module):
    def __init__(self,out_dim,input_channels,length):
        super(WT_embedding, self).__init__()
        ###  下采样 (10,C_out1,1,200)
        C_out1 = 512
        C_out2 = 256
        C_out3 = 128
        down_sample_times = 4
        self.Fra_conv1 = Fractal_inception(input_channel=input_channels, output_channel=C_out1)
        self.down1 = Conv_DownSampling(C_out1) ###  下采样 (10,C_out1,1,100)
        self.conv_mixer1_1 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer1_2 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)

        self.Fra_conv2 = Fractal_inception(input_channel=C_out1, output_channel=C_out2)
        self.down2 = Conv_DownSampling(C_out2)###  下采样 (100,C_out2,1,50)
        self.conv_mixer2_1 = conv_mixer_block(input_channel=C_out2, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer2_2 = conv_mixer_block(input_channel=C_out2, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)

        self.Fra_conv3 = Fractal_inception(input_channel=C_out2, output_channel=C_out3)
        self.down3 = Conv_DownSampling(C_out3)  ###  下采样 (100,C_out3,1,25)
        self.conv_mixer3_1 = conv_mixer_block(input_channel=C_out3, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer3_2 = conv_mixer_block(input_channel=C_out3, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.change_shape1 = nn.Conv2d(C_out3, C_out3, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0))


        self.down4 = Conv_DownSampling(C_out3)  ###  下采样 (100,C_out3,1,12)
        self.conv_mixer4_1 = conv_mixer_block(input_channel=C_out3, conv_mode='inception', kernel_size=[1, 3, 5, 7],
                                              dilated_num=1)
        self.conv_mixer4_2 = conv_mixer_block(input_channel=C_out3, conv_mode='inception', kernel_size=[1, 3, 5, 7],
                                              dilated_num=1)

        vector_len = int(length/(2**down_sample_times))
        encoder_layer = nn.TransformerEncoderLayer(d_model = vector_len,
                                                      nhead = 4,
                                                      dim_feedforward=128,
                                                      dropout=0.1,
                                                      activation='gelu',
                                                      batch_first=True,
                                                      bias=True)
        self.time_embed =  nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(C_out3*vector_len,out_dim)
    def forward(self, x):
        out1 = self.conv_mixer1_2(self.conv_mixer1_1(self.down1(self.Fra_conv1(x) ) ) )
        out2 = self.conv_mixer2_2(self.conv_mixer2_1(self.down2(self.Fra_conv2(out1))))
        out3 = self.change_shape1(self.conv_mixer3_2(self.conv_mixer3_1(self.down3(self.Fra_conv3(out2)))))
        out4 = self.conv_mixer4_2(self.conv_mixer4_1(self.down4(out3)))
        out5 = self.time_embed(out4.squeeze(2))
        out5_flat = out5.reshape(out5.size(0), -1)
        out6 = self.fc1(out5_flat)
        return out6

def WT_embedding_test():
    data = torch.randn(10,256,1,200)
    block = WT_embedding(out_dim = 20,input_channels = 256,length = 200)
    output_tensor = block(data)
    # 打印输出形状
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output_tensor.shape}")

##### wt2 features : (10,48,1,41)
class STFT_embedding(nn.Module):
    def __init__(self,out_dim,input_channels,length):
        super(STFT_embedding, self).__init__()
        self.pre_conv = nn.Conv2d(input_channels, input_channels, kernel_size=(1, 2), stride=(1, 1),padding=(0, 0))
        ###### (10,48,1,40)
        down_sample_times = 2
        C_out1 = 256
        C_out2 = 128
        C_out3 = 64
        ###  下采样 (10,48,1,40)
        self.Fra_conv1 = Fractal_inception(input_channel=input_channels, output_channel=C_out1)
        ###  下采样 (10,C_out1,1,40)
        self.conv_mixer1_1 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer1_2 = conv_mixer_block(input_channel=C_out1, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)

        self.down2 = Conv_DownSampling(C_out1)###  下采样 (100,C_out1,1,20)
        self.Fra_conv2 = Fractal_inception(input_channel=C_out1, output_channel=C_out2)
        self.conv_mixer2_1 = conv_mixer_block(input_channel=C_out2, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer2_2 = conv_mixer_block(input_channel=C_out2, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)

        self.down3 = Conv_DownSampling(C_out2)  ###  下采样 (100,C_out1,1,10)
        self.Fra_conv3 = Fractal_inception(input_channel=C_out2, output_channel=C_out3)
        self.conv_mixer3_1 = conv_mixer_block(input_channel=C_out3, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        self.conv_mixer3_2 = conv_mixer_block(input_channel=C_out3, conv_mode='inception', kernel_size=[1, 3, 5, 7], dilated_num=1)
        vector_len = int(length/(2**down_sample_times))
        encoder_layer = nn.TransformerEncoderLayer(d_model = vector_len,
                                                      nhead = 2,
                                                      dim_feedforward=128,
                                                      dropout=0.1,
                                                      activation='gelu',
                                                      batch_first=True,
                                                      bias=True)
        self.time_embed =  nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(C_out3*vector_len,out_dim)

    def forward(self, x):
        out1 = self.pre_conv(x)
        out1 = self.conv_mixer1_2(self.conv_mixer1_1(self.Fra_conv1(out1) )  )
        out2 = self.conv_mixer2_2(self.conv_mixer2_1(self.Fra_conv2(self.down2(out1))))
        out3 = self.conv_mixer3_2(self.conv_mixer3_1(self.Fra_conv3(self.down3(out2))))
        out4 = self.time_embed(out3.squeeze(2))
        out4_flat = out4.reshape(out4.size(0), -1)
        out5 = self.fc1(out4_flat)
        return out5

def STFT_embedding_test():
    data = torch.randn(10,48,1,41)
    # 设置卷积核大小
    kernel_size = 2

    # 创建卷积层，设置合适的padding
    # conv = nn.Conv2d(9, 9, kernel_size=(1, kernel_size), stride=(1, 1),padding=(0, 0))
    embedding_layer = STFT_embedding(out_dim = 20,input_channels = 48,length = 41)

    output_tensor = embedding_layer(data)
    # 打印输出形状
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output_tensor.shape}")

## signal （100，8，1，200）
class CNN_embedding(nn.Module):
    def __init__(self,out_dim,input_channels,length):
        super(CNN_embedding, self).__init__()
        C_out1 = 32
        C_out2 = 64
        C_out3 = 64
        C_out4 = 64
        down_sample_times = 4
        self.Fra_conv1 = Fractal_inception(input_channel=input_channels, output_channel=C_out1)
        self.down1 = Conv_DownSampling(C_out1)#  下采样 (100,C_out1,1,100)
        self.Fra_conv2 = Fractal_inception(input_channel=C_out1, output_channel=C_out2)
        self.down2 = Conv_DownSampling(C_out2)#  下采样 (100,C_out2,1,50)
        self.Fra_conv3 = Fractal_inception(input_channel=C_out2, output_channel=C_out3)
        self.down3 = Conv_DownSampling(C_out3)#  下采样 (100,C_out3,1,25)
        self.change_shape1 = nn.Conv2d(C_out3, C_out3, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0))
        #  下采样 (100,C_out3,1,24)
        self.Fra_conv4 = Fractal_inception(input_channel=C_out3, output_channel=C_out4)
        self.down4 = Conv_DownSampling(C_out4)  # 下采样 (100,C_out3,1,12)
        self.Fra_conv5 = Fractal_inception(input_channel=C_out4, output_channel=C_out4)
        vector_len = int(length/(2**down_sample_times))
        encoder_layer = nn.TransformerEncoderLayer(d_model = vector_len,
                                                      nhead = 4,
                                                      dim_feedforward=128,
                                                      dropout=0.1,
                                                      activation='gelu',
                                                      batch_first=True,
                                                      bias=True)
        self.time_embed =  nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(C_out4*vector_len,out_dim)

    def forward(self, x):

        out1 = self.down1(self.Fra_conv1(x))
        out2 = self.down2(self.Fra_conv2(out1))
        out3 = self.change_shape1(self.down3(self.Fra_conv3(out2)))
        out4 = self.Fra_conv5(self.down4(self.Fra_conv4(out3)))
        out5 = self.time_embed(out4.squeeze(2))
        out5_flat = out5.reshape(out5.size(0), -1)
        out6 = self.fc1(out5_flat)

        return out6

def CNN_embedding_test():
    data = torch.randn(10,8,1,200)
    embedding_layer = CNN_embedding(out_dim = 20,input_channels = 8,length = 200)
    output_tensor = embedding_layer(data)
    # 打印输出形状
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output_tensor.shape}")
## signal (10,8,1,200)
class CNN_mixers_embedding(nn.Module):
    def __init__(self,out_dim,input_channels,length):
        super(CNN_mixers_embedding, self).__init__()
        depth = 5
        patch_size = 4  # 每个patch的大小
        embed_dim = 64  # 每个patch的嵌入维度
        self.patch = PatchEmbedding(in_channels=input_channels, patch_size=patch_size, embed_dim=embed_dim)
        self.bath1 = nn.BatchNorm2d(embed_dim)
        self.gelu1 = nn.GELU()
        self.conv_mixers = nn.Sequential(
            *[conv_mixer_block(input_channel=embed_dim, conv_mode='norm', kernel_size=3,dilated_num=1) for _ in range(depth)]
        )
        out_H = int(length/patch_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=out_H,
                                                   nhead=5,
                                                   dim_feedforward=128,
                                                   dropout=0.1,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   bias=True)
        self.time_embed = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(out_H*embed_dim, out_dim)

    def forward(self, x):
        out1 = self.gelu1(self.bath1(self.patch(x)))
        out2= self.conv_mixers(out1)
        out3 = self.time_embed(out2.squeeze(2))
        out3_flat = out3.reshape(out3.size(0), -1)
        out4 = self.fc1(out3_flat)

        return out4

def CNN_mixers_embedding_test():
    data = torch.randn(10,8,1,200)
    embedding_layer = CNN_mixers_embedding(out_dim = 20,input_channels = 8,length = 200)
    output_tensor = embedding_layer(data)
    # 打印输出形状
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output_tensor.shape}")

## signal (10,8,1,200)
class CNN_GRU_embedding(nn.Module):
    def __init__(self, out_dim, input_channels, length):
        super(CNN_GRU_embedding, self).__init__()
        down_sample_times = 3
        Cout = 64
        GRU_hidden_size = 64
        GRU_layers = 3
        fc_out1 = 256
        self.conv1 = Inception_ghost(C_in=input_channels, C_out=Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down1 = Conv_DownSampling(Cout)  # 下采样 (100,C_out1,1,100)
        self.conv2 = Inception_ghost(C_in=Cout, C_out=Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down2 = Conv_DownSampling(Cout)  # 下采样 (100,C_out1,1,50)
        self.conv3 = Inception_ghost(C_in=Cout, C_out=Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        self.down3 = Conv_DownSampling(Cout)  # 下采样 (100,C_out1,1,25)
        self.conv4 = Inception_ghost(C_in=Cout, C_out=Cout, kernel_sizes=[1, 3, 5, 7], dilated_num=1)
        input_len = int(length/(2**down_sample_times))
        self.gru = nn.GRU(input_size=input_len, hidden_size=GRU_hidden_size, num_layers=GRU_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(GRU_hidden_size * 2 * Cout, fc_out1)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(fc_out1, out_dim)

    def forward(self, x):
        out1 = self.down1(self.conv1(x))
        out2 = self.down2(self.conv2(out1))
        out3 = self.down3(self.conv3(out2))
        out4 = self.conv4(out3).squeeze(2)
        out5 , _ = self.gru(out4)
        out5 = out5.reshape(out5.size(0), -1)
        out6 = self.fc2(self.gelu1(self.fc1(out5)))
        return out6

def GRU_test():
    rnn = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True)
    input = torch.randn(10, 64, 10)
    output, hn = rnn(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")

def CNN_GRU_test():
    data = torch.randn(10, 8, 1, 200)
    embedding_layer = CNN_GRU_embedding(out_dim=20, input_channels=8, length=200)
    output_tensor = embedding_layer(data)
    # 打印输出形状
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output_tensor.shape}")

import torch.nn.functional as F
def totally_test():
    out_dim = 64
    batch_size = 100
    time_feature_data = torch.randn(batch_size, 24, 30)
    embedd_model_num = 7
    cat_dim = 1
    class_num  = 11
    data_label = torch.randn(batch_size,11)
    # 应用 Softmax 进行归一化，dim=1 表示按行归一化
    data_label = F.softmax(data_label, dim=1)
    TF_embedd = time_feature_embedding(out_dim=out_dim, input_H=24, input_W=30)
    TF_output = TF_embedd(time_feature_data).unsqueeze_(cat_dim)

    F_data = torch.randn(batch_size,2,1,64)
    F_embedd = fre_embedding(out_dim = out_dim,input_channels = 2,length = 64)
    F_output= F_embedd(F_data).unsqueeze_(cat_dim)

    WT_data = torch.randn(batch_size,32,1,256)
    WT_embedd = WT_embedding(out_dim = out_dim,input_channels = 32,length = 256)
    WT_output = WT_embedd(WT_data).unsqueeze_(cat_dim)

    STFT_data = torch.randn(batch_size,9,1,33)
    STFT_embedd = STFT_embedding(out_dim = out_dim,input_channels = 9,length = 33)
    STFT_output = STFT_embedd(STFT_data).unsqueeze_(cat_dim)

    origin_data = torch.randn(batch_size,2,1,128)
    CNN_embedd= CNN_embedding(out_dim = out_dim,input_channels = 2,length = 128)
    CNN_output = CNN_embedd(origin_data).unsqueeze_(cat_dim)

    CNN_mix_embedd = CNN_mixers_embedding(out_dim = out_dim,input_channels = 2,length = 128)
    CNN_mix_output= CNN_mix_embedd(origin_data).unsqueeze_(cat_dim)

    CNN_GRU_embedd = CNN_GRU_embedding(out_dim=out_dim, input_channels=2, length=128)
    CNN_GRU_output= CNN_GRU_embedd(origin_data).unsqueeze_(cat_dim)

    totally_output = torch.cat([TF_output,F_output,WT_output,STFT_output,CNN_output,CNN_mix_output,CNN_GRU_output],dim=cat_dim)


    encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim,
                                               nhead=8,
                                               dim_feedforward=128,
                                               dropout=0.1,
                                               activation='gelu',
                                               batch_first=True,
                                               bias=True)
    feature_embed_layers = nn.TransformerEncoder(encoder_layer, num_layers=6)
    out = feature_embed_layers(totally_output)
    out = out.reshape(out.size(0), -1)

    fc1 = nn.Linear(embedd_model_num * out_dim, class_num)
    gelu1 = nn.GELU()
    out1 = gelu1(fc1(out))
    out2 = fc1(out)

    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(out1, data_label)
    loss2 = criterion(out2, data_label)
    print(f"Loss1: {loss1}")
    print(f"Loss2: {loss2}")

class topo_classify_net(nn.Module):
    def __init__(self, emb_dim, shape_dict,class_num):
        super(topo_classify_net, self).__init__()
        embedd_model_num = 7
        self.cat_dim = 1
        self.TF_embedd = time_feature_embedding(out_dim=emb_dim, input_H=shape_dict['T_H'], input_W=shape_dict['T_W'])
        self.F_embedd = fre_embedding(out_dim=emb_dim, input_channels=shape_dict['F_C'], length=shape_dict['F_Len'])
        self.WT_embedd = WT_embedding(out_dim = emb_dim,input_channels = shape_dict['WT_C'],length = shape_dict['WT_Len'])
        self.STFT_embedd = STFT_embedding(out_dim=emb_dim, input_channels=shape_dict['STFT_C'], length=shape_dict['STFT_Len'])
        self.CNN_embedd = CNN_embedding(out_dim=emb_dim, input_channels=shape_dict['Ori_C'], length=shape_dict['Ori_Len'])
        self.CNN_mix_embedd = CNN_mixers_embedding(out_dim=emb_dim, input_channels=shape_dict['Ori_C'], length=shape_dict['Ori_Len'])
        self.CNN_GRU_embedd = CNN_GRU_embedding(out_dim=emb_dim, input_channels=shape_dict['Ori_C'], length=shape_dict['Ori_Len'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
                                                   nhead=8,
                                                   dim_feedforward=128,
                                                   dropout=0.1,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   bias=True)
        self.feature_embed_layers = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc1 = nn.Linear(embedd_model_num * emb_dim, class_num)
        self.gelu1 = nn.GELU()


    # def forward(self, Data_S):

    def forward(self,Data_S_T_data,Data_S_F_data,Data_S_WT_data,Data_S_STFT_data,Data_S_origin_data):
        TF_output = self.TF_embedd(Data_S_T_data).unsqueeze_(self.cat_dim)
        F_output = self.F_embedd(Data_S_F_data).unsqueeze_(self.cat_dim)
        WT_output = self.WT_embedd(Data_S_WT_data).unsqueeze_(self.cat_dim)
        STFT_output = self.STFT_embedd(Data_S_STFT_data).unsqueeze_(self.cat_dim)
        CNN_output = self.CNN_embedd(Data_S_origin_data).unsqueeze_(self.cat_dim)
        CNN_mix_output = self.CNN_mix_embedd(Data_S_origin_data).unsqueeze_(self.cat_dim)
        CNN_GRU_output = self.CNN_GRU_embedd(Data_S_origin_data).unsqueeze_(self.cat_dim)

        # TF_output = self.TF_embedd(Data_S['T_data']).unsqueeze_(self.cat_dim)
        # F_output = self.F_embedd(Data_S['F_data']).unsqueeze_(self.cat_dim)
        # WT_output = self.WT_embedd(Data_S['WT_data']).unsqueeze_(self.cat_dim)
        # STFT_output = self.STFT_embedd(Data_S['STFT_data']).unsqueeze_(self.cat_dim)
        # CNN_output = self.CNN_embedd(Data_S['origin_data']).unsqueeze_(self.cat_dim)
        # CNN_mix_output = self.CNN_mix_embedd(Data_S['origin_data']).unsqueeze_(self.cat_dim)
        # CNN_GRU_output = self.CNN_GRU_embedd(Data_S['origin_data']).unsqueeze_(self.cat_dim)
        totally_output = torch.cat(
            [TF_output, F_output, WT_output, STFT_output, CNN_output, CNN_mix_output, CNN_GRU_output], dim=self.cat_dim)
        out = self.feature_embed_layers(totally_output)
        out = self.gelu1(self.fc1(out.reshape(out.size(0), -1)))

        return out

def topo_classify_test():
    batch_size = 10  # 假设批次大小为32
    shape_dict = {
        'T_H': 96,  # Time Feature Height (对应 T_data 中的第一个维度)
        'T_W': 30,  # Time Feature Width (对应 T_data 中的第二个维度)
        'F_C': 8,  # Frequency Channels (对应 F_data 中的第一个维度)
        'F_Len': 100,  # Frequency Length (对应 F_data 中的第二个维度)
        'WT_C': 256,  # Wavelet Transform Channels (对应 WT_data 中的第一个维度)
        'WT_Len': 200,  # Wavelet Transform Length (对应 WT_data 中的第二个维度)
        'STFT_C': 48,  # STFT Channels (对应 STFT_data 中的第一个维度)
        'STFT_Len': 41,  # STFT Length (对应 STFT_data 中的第二个维度)
        'Ori_C': 8,  # Original Channels (对应 origin_data 中的第一个维度)
        'Ori_Len': 200  # Original Length (对应 origin_data 中的第二个维度)
    }
    # Data_S 模板
    Data_S = {
        'T_data': torch.randn(batch_size, shape_dict['T_H'], shape_dict['T_W']),    # (batch_size, TF_H, TF_W)
        'F_data': torch.randn(batch_size, shape_dict['F_C'], 1 ,shape_dict['F_Len']),    # (batch_size, F_C, F_Len)
        'WT_data': torch.randn(batch_size, shape_dict['WT_C'], 1,shape_dict['WT_Len']),  # (batch_size, WT_C, WT_Len)
        'STFT_data': torch.randn(batch_size, shape_dict['STFT_C'],1 ,shape_dict['STFT_Len']),  # (batch_size, STFT_C, STFT_Len)
        'origin_data': torch.randn(batch_size, shape_dict['Ori_C'], 1,shape_dict['Ori_Len'])  # (batch_size, Ori_C, Ori_Len)
    }
    my_net = topo_classify_net(emb_dim = 64, shape_dict=shape_dict,class_num = 11)
    output_tensor = my_net(Data_S)

    print(f"Output shape: {output_tensor.shape}")

if __name__ == '__main__':
    # time_feature_embedding_test()
    # fre_embedding_test()
    # WT_embedding_test()
    # STFT_embedding_test()
    # CNN_embedding_test()
    # CNN_mixers_embedding_test()
    # GRU_test()

    # CNN_GRU_test()
    # totally_test()
    topo_classify_test()



