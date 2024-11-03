FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install make git python3 python3-pip -y
RUN python3 -m venv /venv
RUN /venv/bin/pip install -U pip setuptools
RUN /venv/bin/pip install poetry
RUN git clone https://github.com/octakitten/tensorplus.git
WORKDIR /tensorplus
RUN make
RUN /venv/bin/poetry build
