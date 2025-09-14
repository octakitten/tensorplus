FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install bash make git gh python3 python3-pip python3-venv zip curl -y
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN python3 -m venv /venv
RUN /venv/bin/pip install -U pip setuptools pytest coverage pytest-cov
