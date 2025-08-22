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
- Change order (you know what i mean)


Test results for the compressor:
```
config                      fp16 bshuf codecs                   compressed   ratio     time
-------------------------------------------------------------------------------------------
fp32_zstd7                 False False ZSTD_LEVEL_7               19.25 MB   0.876    0.390s
fp16_zstd7                  True False ZSTD_LEVEL_7                8.69 MB   0.396    0.341s
fp32_bshuf_zstd7            True  True ZSTD_LEVEL_7                7.56 MB   0.344    0.326s
fp16_bshuf_zstd7            True  True ZSTD_LEVEL_7                7.56 MB   0.344    0.328s
fp16_bshuf_zstd(3, 7, 22)   True  True ZSTD_LEVEL_3,7,22           7.52 MB   0.342    1.240s
fp16_bshuf_zstd22           True  True ZSTD_LEVEL_22               7.49 MB   0.341    1.154s
raw total:                                                        21.97 MB   1.000         
```