#!/bin/bash

LIB_STDCXX=$(g++ -print-file-name=libstdc++.so)
LIB_ASAN=$(realpath $(gcc -print-file-name=libasan.so))
export LD_PRELOAD="$LIB_ASAN:$LIB_STDCXX"
export ASAN_OPTIONS="detect_leaks=0:log_path=logs/asan_log"
pytest tests/test_meta.py tests/test_dataset.py tests/test_augmentations.py tests/test_compression.py -s --benchmark-disable
# pytest tests/test_meta.py tests/test_dataset.py tests/test_augmentations.py tests/test_compression.py -xs --benchmark-disable
# pytest tests/test_augmentations.py -s
