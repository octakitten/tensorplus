FROM python:3.12

RUN apt-get update
RUN apt-get install -y \
    gcc \
    git \
    vim \
    curl \
    wget \
    unzip
COPY . /tensorplus
WORKDIR /tensorplus
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
RUN make