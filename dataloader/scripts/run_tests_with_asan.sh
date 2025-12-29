export LIB_STDCXX=$(g++ -print-file-name=libstdc++.so)
export LIB_ASAN=$(realpath $(gcc -print-file-name=libasan.so))
export LD_PRELOAD="$LIB_ASAN:$LIB_STDCXX"
echo $LD_PRELOAD
export ASAN_OPTIONS="detect_leaks=0:log_path=logs/asan_log"
pytest tests/test_augmentations.py::TestRasterCorrectness::test_random_crop -s
