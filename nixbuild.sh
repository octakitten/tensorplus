#!/bin/bash
#now deprecated, use build.sh
nix build --impure --extra-experimental-features nix-command \
  --extra-experimental-features flakes
cp result/bin/bin/tensorplus.so src/tensorplus/tensorplus.so
