FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install make git pipx -y
RUN pipx ensurepath
RUN pipx install poetry
RUN git clone https://github.com/octakitten/tensorplus.git
WORKDIR /tensorplus
RUN make
RUN poetry install
RUN poetry build
