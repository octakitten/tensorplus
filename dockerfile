FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install bash make git gh python3 python3-pip python3-venv zip curl python3.12-dev build-essential -y
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN python3 -m venv /venv
RUN /venv/bin/pip install -U pip setuptools pytest coverage pytest-cov
