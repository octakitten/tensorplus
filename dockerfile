FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04


RUN apt-get update && apt-get install bash make git gh gcc-13 g++-13 python3 python3-pip python3-venv zip curl wget python3-dev build-essential -y
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PATH="/usr/bin/gcc-13.2:/usr/bin/g++-13.2:${PATH}"
RUN python3 -m venv /venv
RUN /venv/bin/pip install -U pip setuptools pytest coverage pytest-cov
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
ENV NVM_DIR=/root/.nvm
#SHELL ["/bin/bash", "--login", "-c"]
RUN wget https://raw.githubusercontent.com/nvm-sh/nvm/refs/heads/master/install.sh
RUN bash install.sh && \
    source $NVM_DIR/nvm.sh && \
    nvm install 24
#RUN source /root/.bashrc && nvm install 24
#SHELL ["/bin/bash", "--login", "-c"]
