#!/bin/bash
nix build --impure
cp result/bin/tensorplus.so src/tensorplus/tensorplus.so
cp result/bin/tensorplus.dylib src/tensorplus/tensorplus.dylib
cp result/bin/tensorplus.dll src/tensorplus/tensorplus.dll
poetry build
