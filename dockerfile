FROM nixpkgs/nix-flakes

RUN wget https://github.com/octakitten/tensorplus/archive/refs/heads/main.zip
RUN unzip main.zip
RUN sh tensorplus-main/build.sh

