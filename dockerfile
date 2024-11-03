FROM nixpkgs/nix-flakes

RUN git clone https://github.com/octakitten/tensorplus.git
WORKDIR /tensorplus
RUN nix build --impure
