FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install make git gh python3 python3-pip python3-venv -y
RUN python3 -m venv /venv
RUN /venv/bin/pip install -U pip setuptools
RUN /venv/bin/pip install poetry
