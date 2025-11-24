#!/bin/bash
#now deprecated, use build.sh
#nix shell --impure --extra-experimental-features nix-command --extra-experimental-features flakes nixpkgs.cudaPackages.cuda_nvcc
#nix build --impure --extra-experimental-features nix-command \
#  --extra-experimental-features flakes
#cp -f result/*tensorplus*.so src/tensorplus/tensorplus.so

export NIXPKGS_ALLOW_UNFREE=1
echo "Nix allow unfree: ${NIXPKGS_ALLOW_UNFREE}"
make clean
echo "ran make clean"
make
echo "ran make"
cp tensorplus.so src/tensorplus/tensorplus.so
echo "copied to src"
cp tensorplus.so dist/tensorplus.so
echo "copied to dist"
uv build --wheel
echo "ran uv build"
