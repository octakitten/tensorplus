#!/bin/bash
cp bin/tensorplus.so src/tensorplus/tensorplus.so
cp bin/tensorplus.dylib src/tensorplus/tensorplus.dylib
cp bin/tensorplus.dll src/tensorplus/tensorplus.dll

rm dist/tensorplus.so
rm dist/tensorplus.dylib
rm dist/tensorplus.dll

cp bin/tensorplus.so dist/tensorplus.so
cp bin/tensorplus.dylib dist/tensorplus.dylib
cp bin/tensorplus.dll dist/tensorplus.dll
