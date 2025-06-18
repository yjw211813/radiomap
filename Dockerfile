# 使用CUDA基础镜像
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 设置root密码
RUN echo 'root:123456' | chpasswd

# 备份并更换Ubuntu源
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# 更新系统并安装基础工具
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    python3-pip \
    nano \
    git \
    openssh-server \
    openssh-client

# ================= 安装Miniconda =================
# 下载Miniconda安装脚本 (使用清华镜像源)
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O ~/miniconda.sh

# 安装Miniconda
RUN bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# 将conda加入PATH
ENV PATH=/opt/conda/bin:$PATH

# 配置conda清华源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --set show_channel_urls yes

# ================= 创建conda虚拟环境 =================
RUN conda create -n liaozhengyan python=3.12 -y && \
    echo "conda activate liaozhengyan" >> ~/.bashrc

# 激活环境并配置pip清华源
SHELL ["/bin/bash", "--login", "-c"]
RUN conda init bash && \
    source activate liaozhengyan && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ================= 在虚拟环境中安装PyTorch和依赖 =================
# 方法1：使用pip直接安装PyTorch (推荐)
RUN source activate liaozhengyan && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 方法2：或者使用conda安装PyTorch (二选一)
# RUN source activate liaozhengyan && \
#     conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他Python包
RUN source activate liaozhengyan && \
    pip install --no-cache-dir \
    accelerate \
    einops \
    ema-pytorch \
    pytorch-lightning \
    scikit-learn \
    scipy \
    thop \
    timm \
    tensorboard \
    fvcore \
    albumentations \
    omegaconf \
    numpy

# ================= 配置SSH =================
RUN mkdir -p /run/sshd && \
    chmod 700 /run/sshd && \
    sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# 暴露SSH端口
EXPOSE 22

# 容器启动命令
CMD ["/usr/sbin/sshd", "-D"]