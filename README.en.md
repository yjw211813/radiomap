# radio_map_construction

#### Description
my code

#### Software Architecture
Software architecture description

#### Installation
# 构建镜像 (在Dockerfile目录下执行)
docker build -t liaozhengyan_gpu .
# 运行容器
docker run -d -p 2222:22 --name liaozhengyan_GPU liaozhengyan_gpu
下面是服务器运行时
docker run --gpus all -it --name liaozhengyan_GPU --entrypoint /bin/bash -v /home/cec/student/liaozhengyan/radio_map_construction:/home/code -p 32956:22 liaozhengyan_gpu:latest 
docker run --gpus all -d \
  --name liaozhengyan_GPU \
  -v /home/cec/student/liaozhengyan/radio_map_construction:/home/code \
  -p 32956:22 \
  liaozhengyan_gpu:latest \
  /usr/sbin/sshd -D

docker run -d --name liaozhengyan_GPU -p 32956:22 liaozhengyan_gpu:latest /usr/sbin/sshd -D
docker run --shm-size=32g --gpus all -d --name liaozhengyan_GPU -p 32956:22 liaozhengyan_gpu:latest   /bin/bash -c "/usr/sbin/sshd -D"
ssh root@localhost -p 32956

将修改的容器进行保存
docker commit 842a10df3747 liaozhengyan_gpu:latest
# 进入容器测试
docker exec -it liaozhengyan_GPU bash
docker stop liaozhengyan_GPU && docker rm liaozhengyan_GPU
#### Instructions
wsl --shutdown

diskpart
select vdisk file="C:\Users\Administrator\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
compact vdisk
detach vdisk
exit
https://www.cnblogs.com/cyxg/p/18800234
https://zhuanlan.zhihu.com/p/18333386892

# 容器退出启动容器
docker start liaozhengyan_GPU
docker exec -it liaozhengyan_GPU bash
ps aux | grep sshd
docker run --shm-size=32g --gpus all -d --name liaozhengyan_code -p 32955:22 liaozhengyan_gpu:latest   /bin/bash -c "/usr/sbin/sshd -D"
docker exec -it liaozhengyan_code /bin/bash
print(f"CUDA available: {torch.cuda.is_available()}")

docker run --gpus all -d --name liaozhengyan_GPU -v /home/cec/student/liaozhengyan/radio_map_construction:/home/code -p 32956:22 liaozhengyan_gpu:latest

docker run --shm-size=32g --gpus all -d --name liaozhengyan_GPU -v /home/cecr/liaozhengyan/radio_map_construction:/home/code -p 32956:22 liaozhengyan_gpu:latest
# 输入输出的形状
条件相关的形状为

条件的形状(4,256,256)
生成目标形状(1,256,256)
原始输入(1,256,256)
## 三个可能的改进方向
先对下面内容进行改进
- 中间网络层 注意力模块的改进
- 条件网络改进 
- 残差模块改进   √
- 条件引入改进   
- 原始输入改进 （简单插值 单独网络生成 ）
 时间向量应该如何引入到当前的网络中呢？
- 
相似内积模块设计

- 中间层加入边缘检测模块
- LOS引入
- 相关射线传播规律 网络按照物理功能分解
- 引入物理信息相关损失
- 扩散模型框架加速
sudo ./cuda_12.2.2_535.104.05_linux.run 

sudo adduser jinshidong
sudo usermod -aG docker jinshidong

然后使用pinns进行embedding
利用fft来辅助构造loss函数


pip install --no-index --find-links=./spikingjelly_offline_pkgs spikingjelly

pip install h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

docker run --shm-size=8g -d --name liaozhengyan_GPU -p 32956:22 liaozhengyan_gpu:latest
docker run --shm-size=64g --gpus all -d --name liaozhengyan_GPU -v ~/data:/home/data -v ~/code:/home/code -p 32956:22 liaozhengyan_gpu:latestV2
docker run --shm-size=64g --gpus all -d --name liaozhengyan_GPU -v ~/dataset:/home/data -v ~/code:/home/code -p 32956:22 liaozhengyan_gpu:latestV2
docker run --shm-size=64g --gpus all -d --name liaozhengyan_GPU -v ~/liaozhengyan/dataset:/home/data -v ~/liaozhengyan/code:/home/code -p 32944:22 liaozhengyan_gpu:latestV2

docker run --shm-size=64g --gpus all -d --name liaozhengyan_mamba -v ~/liaozhengyan/dataset:/home/data -v ~/liaozhengyan/code:/home/code -p 32944:22 liaozhengyan_gpu:latest_mamba_xiao

ls -ld ~/liaozhengyan
sudo chown -R liaozhengyan:liaozhengyan ~/liaozhengyan


# 7月18号 ToDoList
1.重构第一篇小论文的代码
2.引入新的模型和数据集进行算法验证
3.做第一篇小论文的汇报ppt


4.开始写小论文的文字稿部分
可以开始写小论文相关介绍部分和贡献部分
4.将现有网络引入到当前的DDPM,DDIM,RFLOW中去
5.然后引入到 EDM和DPM-solver

整合第一篇小论文的算法流程描述
现有算法步骤（如算法2、3、4）的伪代码表述较繁琐，可合并重复逻辑（如列正交化与QR分解的协同过程），并补充关键步骤的物理意义说明（如谱截断为何能抑制噪声敏感性），避免纯数学推导导致的理解障碍。
统一公式中符号大小写（如“H(X)”与“H”混用），并为图2(a)-(c)添加坐标轴单位（如“矩阵阶数”“RMSE”）；建议增加不同算法在数据集上的收敛曲线对比，直观展示QR-ILSM的迭代效率优势。

完善特色四代码的应用逻辑
py2001进行计算
信号 传播 信噪比计算 

v1 [1,3,5,7] 卷积核
val avg_nmse_loss: 0.0064
val Loss: 0.0039
val avg_ssim_loss: 0.9237
val avg_psnr_loss: 34.3554

v2 [3,5,7,9] 增大感受野 现在最优
val avg_nmse_loss: 0.0068
val Loss: 0.0041
val avg_ssim_loss: 0.9324
val avg_psnr_loss: 34.3426
只是精度更为稳定，最高性能提升并不是很大

v3 [3,5,7,9] 空洞卷积加 残差模块
val avg_nmse_loss: 0.0085
val Loss: 0.0044
val avg_ssim_loss: 0.9354
val avg_psnr_loss: 33.1609
v4 [3,5,7,9] 增大感受野 + 残差模块

val NMSE: 0.0095
val RMSE: 0.0231
val SSIM: 0.9408
val PSNR: 32.6419

multi_scale v5 将多尺度卷积核加入到注意力核和 Unet卷积网络中没有动中间的分形卷积
Epoch [947/2000], Train Loss: 0.0000
val NMSE: 0.0060
val RMSE: 0.0183
val SSIM: 0.9158
val PSNR: 34.6274

multi_scale v6 将多尺度卷积核加入到注意力核和 Unet卷积网络 全部替换成 多尺度卷积
Epoch [779/2000], Train Loss: 0.0000
val NMSE: 0.0052
val RMSE: 0.0171
val SSIM: 0.9325
val PSNR: 35.2647
Epoch [959/2000], Train Loss: 0.0000
val NMSE: 0.0050
val RMSE: 0.0167
val SSIM: 0.9194
val PSNR: 35.4381
ghost v6 在原有v1的基础上将网络加深
Epoch [491/2000], Train Loss: 0.0000
val NMSE: 0.0069
val RMSE: 0.0197
val SSIM: 0.9254
val PSNR: 33.9588

multi_scale v7 加入 SSIM loss到训练中：
Epoch [209/2000], Train Loss: 0.0001
val NMSE: 0.0058
val RMSE: 0.0180
val SSIM: 0.8694
val PSNR: 34.8455

multi_scale v8 降低 SSIM loss对整体loss的影响
好像图像过于平滑了

multi_scale v9 降低 SSIM loss对整体loss的影响 在8的基础上引入随机loss
8的效果训练过程不太稳定
然后

Epoch [682/2000], Train Loss: 0.0000
val NMSE: 0.0060
val RMSE: 0.0183
val SSIM: 0.9278
val PSNR: 34.6751
Epoch [569/2000], Train Loss: 0.0000
val NMSE: 0.0059
val RMSE: 0.0182
val SSIM: 0.9279
val PSNR: 34.7340

multi_scale v10 使用对数化的 傅里叶loss来训练模型
val NMSE: 0.0054
val RMSE: 0.0175
val SSIM: 0.9374
val PSNR: 35.0713

Epoch [11/2000], Train Loss: 0.0014
val NMSE: 0.0067
val RMSE: 0.0194
val SSIM: 0.9464
val PSNR: 34.0944


multi_scale v11 增加傅里叶高频区域loss
val NMSE: 0.0052
val RMSE: 0.0171
val SSIM: 0.9338
val PSNR: 35.2107

Epoch [12/2000], Train Loss: 0.0005
val NMSE: 0.0064
val RMSE: 0.0189
val SSIM: 0.9507
val PSNR: 34.3358




无人机 采购 到货
无人机试飞做实验
将实验步骤设计清楚
无人机如何操作？
特色无人机实验
工程学术研讨会



docker run --shm-size=16g --gpus all -d --name deep_au_test  deep_au:v1



echo "export PYTHONPATH=\$PYTHONPATH:/home/code/radio_map_construction" >> ~/.bashrc
source ~/.bashrc
python3 /home/code/radio_map_construction/train/model_train_multiscale_v8.py


