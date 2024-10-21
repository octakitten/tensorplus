cd nix
nix build --impure
nix-shell
cd ..
poetry build
source ./.venv/bin/activate
pip install dist/*.whl --force-reinstall