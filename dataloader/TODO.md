Memory bandwidth utilization in GB/s on RTX 4080 system:
- JAX (default): 7.25
- CUDA (pinned): 19.41
- CUDA (malloc): 9.07

TODO (important improvements for production use):
- Add support for compressed files in the dataloader
- Cleanup all data options
- Fix last races and improve test suite
- Support WebDataset
- Dataset refactor with probing/autodiscovery
- Probing should happen after the augmentations have been applied... Idk??
- Look at any race conditions chadgpt might have produced during the last refactor
- Fix regression where subdir order does not matter.

Fixed:
- Fix all sensible clang-tidy warnings
- Write tests for different prefetch size, thread size configurations
- Deal with parallel dataloaders (as that doesn't really make much sense)
- Replace existing build system with something much faster and easier to debug
- llm code context

Feature List (- not begun, w working, + solved):
    I/O:
    - tests for slow drives with custom fuse filesystem
    - overlapped i/o

    Application:
    - training with synthetic data
    - document full pipeline for calib, train, inference, post-process
    - solve for ground plane like in zhang method, masks have to be respected

    Benchmark/Perf.:
    - perf counters (with python apis)
    - benchmark has to document theoretical max values for all pipline stages
    - autotune option for dataloader (i.e. minimize #threads while keeping within an inch of max throughput.)

    Augmentations:
    + make pipeline work; only bilinear resize for now
    + vectorisation api
    + tests for augmentations
    w test augmentations with inconsistent input shapes

    Compression:
    - enable compressed file format use in dataloader

    Misc:
    + ensure easy setup with uv (both production and tests)
    + apply more sanitizers
    + proper memory arenas in all perf. critical areas; static memory footprint as far as possible in data sources, decoders, augmentation pipe, decompression and resource pool.
    - zero-copy augmentation output to pinned memory
    + fix item keys order
    + graphics card backend (e.g. cuda) selectable at runtime (default to cuda though)
    - enable clang-tidy for all const modernisations
    + support jax and pytorch bindings for IDataLoader out of the box; no tensorflow, because me not like it
    + thread exception exits the program


Plan for how to select the backend:
Needs to happen before the resource pool is initialized, as you obviously cannot switch as soon as the resource pool is actually in use.

TODO (IMPORTANT IF THE DATALOADER IS EVER SUPPOSED TO WORK; THINK ABOUT GENCHANGE): setBuffers for the augmentation pipe

TODO (Document in README later): Enable address sanitizer to work in dbeug mode using LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS="detect_leaks=0:log_path=logs/asan_log".

TODO (Run tests using): uv run --group test pytest ./tests/test_dataloader.py

Tutorial (How to deal with debug-asan):
make install-debug-asan
LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS="detect_leaks=0:log_path=logs/asan_log" pytest tests/...
