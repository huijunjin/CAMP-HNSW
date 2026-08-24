#!/usr/bin/env bash
#
# rebuild_hnswlib.sh -- clean rebuild of the hnswlib fork after editing
# hnswlib/hnswalg.h or python_bindings/bindings.cpp:
#   1) Remove the root build/ directory, then CMake + make -j
#   2) Remove stale build/dist/*.egg-info under python_bindings/
#   3) Uninstall any previously installed hnswlib
#   4) Reinstall in editable mode (pip install -e .)

set -e  # exit immediately on error

ROOT_DIR=$(pwd)

echo "=== (1) Remove main build/ folder ==="
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build

echo "=== (2) CMake & make in main hnswlib folder ==="
cmake ..
make -j

echo "=== (3) Clean up python_bindings old build/dist ==="
cd "$ROOT_DIR/python_bindings"

rm -f hnswlib/*.so *.so

if [ -d build ]; then
    rm -rf build
fi
if [ -d dist ]; then
    rm -rf dist
fi
if [ -d hnswlib.egg-info ]; then
    rm -rf hnswlib.egg-info
fi

echo "=== (4) Uninstall old hnswlib from site-packages ==="
pip uninstall -y hnswlib || true  # OK if it wasn't installed

echo "=== (5) Re-install (editable mode) python bindings ==="
pip install --no-build-isolation -e .

echo "=== Done! ==="
