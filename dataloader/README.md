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
- Upload uint8 array to gpu and cast manually? Supporting exr files too means it needs to deal with both float32 and uint8 and cast appropriately in jax, if necessary.
- Connect concept of batch size to the dataset somehow, as it cannot change during use anyways and that just confuses things.
