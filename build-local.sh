make
rm src/tensorplus/tensorplus.so
cp tensorplus.so src/tensorplus/tensorplus.so
.venv/bin/python3 -m uv build --wheel
