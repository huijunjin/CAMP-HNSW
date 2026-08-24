#!/usr/bin/env bash
#
# build_hnswlib.sh
#   1) 루트 build/ 폴더 삭제 -> CMake + make -j
#   2) python_bindings 내 build/dist/*.egg-info도 삭제
#   3) pip uninstall hnswlib (중복 제거)
#   4) pip install -e .
#   => 수정사항 반영된 상태로 Jupyter에서 import 가능.

set -e  # 에러 발생 시 스크립트 즉시 종료

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

# 만약 build, dist, egg-info 폴더가 있다면 삭제
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
pip uninstall -y hnswlib || true  # 혹시 install 안된 상태라도 OK

echo "=== (5) Re-install (editable mode) python bindings ==="
pip install --no-build-isolation -e .

echo "=== Done! ==="
