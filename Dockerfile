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
RUN pip install -U MinkowskiEngine --install-option="--blas=openblas" --install-option="--force_cuda" -v --no-deps 

COPY . ${HOME}