FROM nixos/nix

RUN wget https://github.com/octakitten/tensorplus/archive/refs/heads/main.zip
RUN gzip main.zip
RUN bash build.sh
