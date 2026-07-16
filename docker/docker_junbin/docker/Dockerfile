# ======================== 以下是你已经缓存好的部分 (保持原样) ========================
# 1. 使用 Nvidia 官方的 CUDA 12.6 + Ubuntu 22.04 基础镜像
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

# 2. 避免 apt 安装时弹出时区选择等交互界面
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# 3. 安装基础系统工具和 Python 3.10
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. 设置 ROS 2 Humble 的 apt 仓库并安装原生 ROS 2
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# 5. 初始化 rosdep (ROS 包依赖管理)
RUN rosdep init && rosdep update

# 6. 配置 Python pip，并安装 requirements.txt 里的包
WORKDIR /tmp

# 7. 严格对齐核心 AI 算力栈 (匹配 CUDA 12.6 与 Torch 2.7.1)
RUN pip3 install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126

# 安装 xformers
RUN pip3 install --no-cache-dir xformers==0.0.31

# 安装 torch_scatter
RUN pip3 install --no-cache-dir torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.1+cu126.html

# 8. 如果需要 Nvidia 的 RAPIDS，从 Nvidia 官方源安装
RUN pip3 install --no-cache-dir cudf-cu12 cuml-cu12 rmm-cu12 --extra-index-url=https://pypi.nvidia.com

# 10. 临时以 root 身份补齐编译 OpenVDB 必需的底层库
# 虽然写在中间不美观，但这是为了不破坏你前面的 cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake sudo libboost-iostreams-dev libtbb-dev libblosc-dev \
    && rm -rf /var/lib/apt/lists/*

# 11. 编译安装补丁版 OpenVDB (支持 Int8Grid)
WORKDIR /tmp
RUN pip3 install --no-cache-dir nanobind
RUN apt-get update && apt-get install -y cmake
RUN wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz && \
    tar xzvf boost_1_86_0.tar.gz && cd boost_1_86_0 && ./bootstrap.sh --prefix=/usr && ./b2 && ./b2 install && \
    rm -rf /tmp/boost*

RUN git clone https://github.com/OasisArtisan/openvdb && \
    mkdir openvdb/build && cd openvdb/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DOPENVDB_BUILD_PYTHON_MODULE=ON -DOPENVDB_BUILD_PYTHON_UNITTESTS=ON \
      -DOPENVDB_PYTHON_WRAP_ALL_GRID_TYPES=ON -DUSE_NUMPY=ON \
      -Dnanobind_DIR=$(python3 -c "import nanobind; print(nanobind.cmake_dir())") .. && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/openvdb

# 9. 安装剩余的通用算法包 (包含 AirSim 补丁)
COPY requirements.txt .
RUN pip3 install --no-cache-dir msgpack-rpc-python
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir --force-reinstall --no-deps "protobuf==3.20.3"
RUN pip3 install --no-cache-dir --no-deps "rerun-sdk==0.24.1"

RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
    
# 12. 用户映射与权限管理
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} dev && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash dev && \
    usermod -aG sudo dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 13. 切换为 dev 用户，后续所有环境配置都归属于他
USER dev
ENV HOME=/home/dev
RUN rosdep update
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 14. 最终的工作目录 (与 compose 挂载点对齐)
WORKDIR /RayFronts
CMD ["bash"]