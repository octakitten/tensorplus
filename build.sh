nix build --impure
poetry build
source ./.venv/bin/activate
pip install dist/*.whl --force-reinstall
