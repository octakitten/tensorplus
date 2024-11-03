FROM ubuntu

RUN apt-get update
RUN apt-get install curl -y
RUN <(curl -L https://nixos.org/nix/install) --no-daemon -y
RUN curl https://github.com/octakitten/tensorplus/archive/refs/heads/main.zip
RUN gzip -d main.zip
RUN sh build.sh
