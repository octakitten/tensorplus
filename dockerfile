FROM ubuntu

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install curl wget unzip -y
RUN curl https://nixos.org/nix/install | sh
RUN wget https://github.com/octakitten/tensorplus/archive/refs/heads/main.zip
RUN unzip main.zip
RUN build.sh
