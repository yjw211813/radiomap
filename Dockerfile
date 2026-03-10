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


# 更新系统并安装基础工具
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    python3-pip \
    python3-dev \
    nano \
    git \
    openssh-server \
    openssh-client \
    # 添加matplotlib需要的系统依赖
    libgl1-mesa-glx \
    libglib2.0-0

# 配置pip清华源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ================= 安装PyTorch和依赖 =================
# 使用pip直接安装PyTorch
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 安装其他Python包
RUN pip3 install --no-cache-dir \
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
    numpy \
    pandas \
    scikit-image \
    matplotlib \
    wandb \
    torchsummary

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