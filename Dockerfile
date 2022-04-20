FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
ENV HOME=/opt
# ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${HOME}
COPY . ${HOME}
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractivve TZ=Asia/Shanghai apt-get -y install tzdata
RUN apt-get install --no-install-recommends -y \
    libgl1 libglib2.0 libgomp1