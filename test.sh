#!/bin/bash
nix develop --impure --command 'source ./.venv/bin/activate'
nix develop --impure --command 'pip install dist/*.whl --force-reinstall'
nix develop --impure --command 'python3 test/test_tensorplus.py'

