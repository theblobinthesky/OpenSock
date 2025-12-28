export LD_PRELOAD=$(gcc -print-file-name=libasan.so) 
export ASAN_OPTIONS="detect_leaks=0:log_path=logs/asan_log"
pytest tests/test_augmentations.py::TestBasics::test_augment_raster_executes_successfully
