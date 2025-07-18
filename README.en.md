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
- 残差模块改进
- 条件引入改进
- 原始输入改进 （简单插值 单独网络生成 ）

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
docker run --shm-size=64g --gpus all -d --name liaozhengyan_GPU -v ~/liaozhengyan/dataset:/home/data -v ~/liaozhengyan/code:/home/code -p 32956:22 liaozhengyan_gpu:latestV2
ls -ld ~/liaozhengyan
sudo chown -R liaozhengyan:liaozhengyan ~/liaozhengyan






