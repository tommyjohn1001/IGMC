FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV HTTP_PROXY=http://proxytc.vingroup.net:9090/ 
ENV HTTPS_PROXY=http://proxytc.vingroup.net:9090/
ENV http_proxy=http://proxytc.vingroup.net:9090/
ENV https_proxy=http://proxytc.vingroup.net:9090/

ENV PATH="/usr/local/miniconda3/bin:${PATH}"
ARG PATH="/usr/local/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh \
    && mkdir /usr/local/.conda \
    && bash Miniconda3-py38_4.11.0-Linux-x86_64.sh -b -p /usr/local/miniconda3\
    && rm -f Miniconda3-py38_4.11.0-Linux-x86_64.sh

RUN conda install -y pytorch=1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
COPY torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl .
RUN pip install ./torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
COPY torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl .
RUN pip install ./torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
COPY additional_requirements.txt .
RUN pip install -r additional_requirements.txt
ENTRYPOINT /bin/bash -c