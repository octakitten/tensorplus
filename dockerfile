FROM ubuntu

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install curl -y
RUN curl https://nixos.org/nix/install | sh --no-daemon -y
RUN curl https://github.com/octakitten/tensorplus/archive/refs/heads/main.zip
RUN unzip main.zip
RUN build.sh
