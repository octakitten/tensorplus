make
rm src/tensorplus/tps.so
mv tps.so src/tensorplus/tps.so
uv build --wheel
