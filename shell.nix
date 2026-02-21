let
  pkgs = import <nixpkgs> {};
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
