FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV TORCH_CUDA_ARCH_LIST="7.0"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Git config
RUN git config --global  http.proxy $http_proxy
RUN git config --global  https.proxy $https_proxy


# Install APEX
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/apex && cd apex && git reset --hard b88c507edb0d067d5570f7a8efe03a90664a3d16
RUN pip install --upgrade pip
RUN pip install packaging
RUN pip uninstall apex
RUN cd apex && TORCH_CUDA_ARCH_LIST="7.0" pip install --upgrade --force-reinstall -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


# Install Paws
WORKDIR /workspace/paws
COPY . .
RUN pip3 install -r requirements.txt --ignore-installed PyYAML
# Install notebook
RUN python3 -m pip install jupyter

RUN jupyter notebook --generate-config

