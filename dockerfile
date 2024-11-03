FROM nixos/nix

RUN curl https://github.com/octakitten/tensorplus/archive/refs/heads/main.zip
RUN gzip -d main.sip
