FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04 AS CUDA

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime AS TORCH_BASE
ENV HOME=/opt
# ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${HOME}
RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractivve TZ=Asia/Shanghai apt-get -y install tzdata
RUN apt-get install --no-install-recommends -y \
    libgl1 libglib2.0 libgomp1 build-essential python3-dev libopenblas-dev

COPY requirements.txt ${HOME}
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
RUN pip install numpy ninja -i https://mirrors.aliyun.com/pypi/simple

COPY --from=CUDA /usr/local/cuda /usr/local/cuda
COPY MinkowskiEngine/ ${HOME}/MinkowskiEngine
WORKDIR ${HOME}/MinkowskiEngine/
RUN TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr/local/cuda python setup.py install --blas=openblas --force_cuda

RUN pip install imageio -i https://mirrors.aliyun.com/pypi/simple
WORKDIR ${HOME}
# COPY . ${HOME}