#!/bin/bash
nix build --impure --extra-experimental-features nix-command \
  --extra-experimental-features flakes
cp result/bin/bin/tensorplus.so src/tensorplus/tensorplus.so
cp result/bin/bin/tensorplus.dylib src/tensorplus/tensorplus.dylib
cp result/bin/bin/tensorplus.dll src/tensorplus/tensorplus.dll
