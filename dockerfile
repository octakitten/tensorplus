FROM gcc:bookworm

RUN apt-get update
RUN apt-get install -y \
    git \
    vim \
    curl \
    wget \
    unzip
RUN wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-debian12-12-5-local_12.5.0-555.42.02-1_amd64.deb
RUN dpkg -i cuda-repo-debian12-12-5-local_12.5.0-555.42.02-1_amd64.deb
RUN mkdir -p /usr/share/keyrings
RUN cp /var/cuda-repo-debian12-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-5
COPY . /tensorplus
WORKDIR /tensorplus
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
RUN make