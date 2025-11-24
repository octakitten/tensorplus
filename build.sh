make
rm src/tensorplus/tensorplus.so
mv tensorplus.so src/tensorplus/tensorplus.so
uv build --wheel
