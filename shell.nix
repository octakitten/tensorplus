let
  pkgs = import (buildins.fetchGit {
    name = "nixos-unstable";
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "refs/heads/nixpkgs-unstable";
    rev = "e6f23dc08d3624daab7094b701aa3954923c6bbb";
  }) {};
in pkgs.mkShell {
  packages = [
  pkgs.python313
  pkgs.uv
  pkgs.gnumake
  pkgs.cudaPackages.cudatoolkit
  pkgs.gcc13
  pkgs.gcc13Stdenv
  pkgs.bash
  pkgs.zip
  pkgs.linuxPackages.nvidia_x11
  ];
  shellHook = ''
    export NIXPKGS_ALLOW_UNFREE=1
    make
    rm src/tensorplus/tensorplus.sh
    mv tensorplus.so src/tensorplus/tensorplus.so
    uv sync
    uv build --wheel
    '';
  }
