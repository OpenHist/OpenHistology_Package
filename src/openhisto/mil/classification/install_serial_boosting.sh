#!/usr/bin/env bash
# Build the tested CPU backends inside the supplied virtual environment.
set -euo pipefail
head_python="${1:-python3}"
"$head_python" -c 'import sys; assert sys.prefix != sys.base_prefix, "Activate a Python virtual environment before running this script"'
"$head_python" -m pip install cmake==4.4.4 ninja==1.13.2 scikit-build-core==1.1.1
CMAKE_BUILD_PARALLEL_LEVEL="${CPATH_HEAD_BUILD_JOBS:-4}" "$head_python" -m pip install --no-cache-dir --no-deps --force-reinstall --no-binary=xgboost --config-settings=cmake.define.USE_OPENMP=OFF xgboost==3.4.1
CMAKE_BUILD_PARALLEL_LEVEL="${CPATH_HEAD_BUILD_JOBS:-4}" "$head_python" -m pip install --no-cache-dir --no-deps --force-reinstall --no-binary=lightgbm --config-settings=cmake.define.USE_OPENMP=OFF lightgbm==4.7.0
"$head_python" -c 'import xgboost, lightgbm; print("Serial CPU imports:", xgboost.__version__, lightgbm.__version__)'
