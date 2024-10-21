nix build --impure
nix-shell
poetry build
source ./.venv/bin/activate
pip install dist/*.whl --force-reinstall