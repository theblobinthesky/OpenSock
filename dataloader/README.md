Memory bandwidth utilization in GB/s on RTX 4080 system:
- JAX (default): 7.25
- CUDA (pinned): 19.41
- CUDA (malloc): 9.07

Benchmark results in seconds:
- JAX put_device: 3.61
- Custom pinned: 5.15 
- Custom pinned + avx2: 14.13
I broke something midway...

Meh.. Damn.

TODO (important improvements for production use):
- Add processing list to data loader in order to remove reset race condition
- Fix tests (they are acktschually incorrect)
- Fix memcpy race condition
