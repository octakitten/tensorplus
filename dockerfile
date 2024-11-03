FROM ubuntu

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install curl wget unzip adduser -y
RUN adduser --disabled-password --gecos '' dumbo
RUN mkdir -m 0755 /nix && chown dumbo /nix
USER dumbo
ENV USER dumbo
WORKDIR /home/dumbo
RUN curl https://nixos.org/nix/install | sh
RUN . .nix-profile/etc/profile.d/nix.sh && nix-channel --update
RUN wget https://github.com/octakitten/tensorplus/archive/refs/heads/main.zip
RUN unzip main.zip
WORKDIR /home/dumbo/tensorplus-main
RUN . .nix-profile/etc/profile.d/nix.sh && nix build 
RUN sh build.sh

