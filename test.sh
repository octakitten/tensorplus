#!/bin/bash
./testvenv/bin/pip install pytest
./testvenv/bin/pip install dist/*.whl --force-reinstall
./testvenv/bin/python3 -m pytest test
