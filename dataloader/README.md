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
- Refactor threadpool for each individual dataloader
- Change order (you know what i mean)


Test results for the compressor:
```
config                      fp16 bshuf codecs                   compressed   ratio  comp(s)  c_in MB/s  c_out MB/s   dec(s)  d_in MB/s  d_out MB/s
--------------------------------------------------------------------------------------------------------------------------------------------------
fp32_zstd7                 False False ZSTD_LEVEL_7              615.17 MB   0.875    6.348     110.76       96.90    2.703     227.56      260.09
fp16_zstd7                  True False ZSTD_LEVEL_7              275.40 MB   0.392    5.398     130.26       51.02    2.433     113.19      144.49
fp32_bshuf_zstd7           False  True ZSTD_LEVEL_7              516.59 MB   0.735    6.084     115.56       84.90    2.742     188.42      256.45
fp16_bshuf_zstd7            True  True ZSTD_LEVEL_7              238.41 MB   0.339    5.154     136.41       46.25    2.436      97.88      144.33
fp16_bshuf_zstd(3,7,22)     True  True ZSTD_LEVEL_3,7,22         236.77 MB   0.337   20.931      33.59       11.31    2.464      96.08      142.66
fp16_bshuf_zstd22           True  True ZSTD_LEVEL_22             235.98 MB   0.336   20.030      35.10       11.78    2.494      94.62      140.97
raw total:                                                       703.13 MB   1.000
```
