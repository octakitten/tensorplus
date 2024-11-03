FROM nixos/nix

RUN wget https://github.com/octakitten/tensorplus/archive/refs/heads/main.zip
RUN gzip -d main.zip
RUN bash tensorplus/build.sh
