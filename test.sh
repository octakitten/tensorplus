!#/bin/zsh
make clean
make
poetry build
source ./.venv/bin/activate
pip install dist/*.whl --force-reinstall
python3 test/test_tensorplus.py
