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

v1 到 v2 就是从重排上采样到反卷积上采样
v2 到 v3 就是将原来的inception变成残差模块，加大了卷积核大小，并且加了空洞
v3 到 v4 则是去掉其中的空洞喝扩大卷积核 只用残差模块
v4 到 v6 则是将网络加深 然后还是使用元素重排上采样
v4 到 multi_v5 将其中残差卷积块换成了多尺度卷积块
multi_v5 到 multi_v6 则是将中间残差分形卷积换成不是残差的分形卷积

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

echo "export PYTHONPATH=\$PYTHONPATH:/home/code/radioMap" >> ~/.bashrc
source ~/.bashrc
python3 /home/code/radioMap/train/Unet_BTM_train.py

# 出现的问题
加入傅里叶损失和小波损失
1.出现频域和小波域的幅度对应不上
2.两个损失非线性程度太高，不收敛

# 现在发现应该采用学习率调度策略来对模型进行调优
并且应该赶紧实现一些基准模型
然后进行实验对比
当前目标数据预处理部分
1、需要补充同一样本反复采样的数据
2、需要多进行几组数据预处理工作（使用一些经典插值算法）
当前模型结构采样问题
1、需要加入一些先进的通道注意力

2、需要引入UNet++的结构

3、需要一些空间注意力（尽量将模块本身的作用想明白）

4、引入一些编码误差修正过程

算法后处理部分可以加一个修正模块进行高频部分的修正
1、看是否有哪些高频注意力机制可以使用

# flowmatching model 
条件训练的代码需要写一个if else
采样过程需要将求解算法和速度预测分开
需要将DDPM扩散过程进行对象化

flowmatching setting
------------------------------
flowmatching APP.py  volecity_predict.py
---------------------------
scheduler.py


set PYTHONPATH=C:\Users\Administrator\Desktop\notebook\second_paper\radio_map_construction

set PYTHONPATH=C:\Users\Administrator\Desktop\radiomap\radio_map_construction

set PYTHONPATH=/home/code/radioMap

lightning 框架