#!/bin/bash
#nix build --impure
#nix-shell --impure
mv -f tensorplus.so src/tensorplus/tensorplus.so
mv -f tensorplus.dylib src/tensorplus/tensorplus.dylib
mv -f tensorplus.dll src/tensorplus/tensorplus.dll
poetry build
