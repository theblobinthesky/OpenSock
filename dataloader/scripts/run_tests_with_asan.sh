LIB_STDCXX=$(g++ -print-file-name=libstdc++.so)
LIB_ASAN=$(realpath $(gcc -print-file-name=libasan.so))
export LD_PRELOAD="$LIB_ASAN:$LIB_STDCXX"
export ASAN_OPTIONS="detect_leaks=0:log_path=logs/asan_log"
# pytest tests/test_augmentations.py -s
pytest tests/test_augmentations.py::TestPointCorrectness::test_flip -s
# pytest tests/test_augmentations.py::TestRasterCorrectness -s
