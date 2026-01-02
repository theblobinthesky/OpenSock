# Benchmark Results

Generated: 2026-01-02 14:50:13

Command: `pytest ./tests --benchmark-only`

## System

- platform: Linux-6.8.0-90-generic-x86_64-with-glibc2.39
- python: 3.12.3
- cpu: 13th Gen Intel(R) Core(TM) i7-13700K
- cpu_count: 24
- mem_total: 32597212 kB
- gpu: NVIDIA GeForce RTX 4080, 565.57.01, 16376 MiB

## Compression / test_compress[fp32_zstd7]

- mean: 3.6583642516670807
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 192.19868884832593
  - c_out_MBps: 168.15441619848045
  - ratio: 0.874898872651415

## Compression / test_compress[fp16_zstd7]

- mean: 2.3307127566683143
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 301.68145366188656
  - c_out_MBps: 118.16041861322738
  - ratio: 0.39167279651753867

## Compression / test_compress[fp32_bshuf_zstd7]

- mean: 3.09044976799972
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 227.51795540592127
  - c_out_MBps: 167.15746570215072
  - ratio: 0.7347001048946679

## Compression / test_compress[fp16_bshuf_zstd7]

- mean: 2.187434799001494
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 321.44172380404734
  - c_out_MBps: 108.99199723899032
  - ratio: 0.33907233930039665

## Compression / test_compress[fp16_bshuf_zstd(3,7,22)]

- mean: 8.48677635599961
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 82.85039961055766
  - c_out_MBps: 27.89889143883204
  - ratio: 0.3367381638467906

## Compression / test_compress[fp16_bshuf_zstd22]

- mean: 8.0332203216664
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 87.52813745237141
  - c_out_MBps: 29.375299914173763
  - ratio: 0.33560979096760035

## Decompression / test_decompress[fp32_zstd7]

- mean: 2.8896254043332497
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 243.3300909680507
  - d_out_MBps: 243.32738733041373

## Decompression / test_decompress[fp16_zstd7]

- mean: 2.6591122850004467
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 264.4238893055552
  - d_out_MBps: 132.21047564749261

## Decompression / test_decompress[fp32_bshuf_zstd7]

- mean: 2.880982992332671
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 244.0600358875039
  - d_out_MBps: 244.05732413945788

## Decompression / test_decompress[fp16_bshuf_zstd7]

- mean: 2.6120606400011943
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 269.1870172277771
  - d_out_MBps: 134.59201314707582

## Decompression / test_decompress[fp16_bshuf_zstd(3,7,22)]

- mean: 2.654814629999843
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 264.85194278895534
  - d_out_MBps: 132.4245000111442

## Decompression / test_decompress[fp16_bshuf_zstd22]

- mean: 2.6565362106654598
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 264.6803042537357
  - d_out_MBps: 132.33868169707122

## Dataloader / test_end_to_end_perf

- mean: 0.07450499366556566
- rounds: 3
- iterations: 1
- extra:
  - batches: 32
  - in_MBps: 352.8624327021994
  - items: 138240000
  - out_MBps: 7077.965167906927

## Raw Output

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0 -- /data/OpenSock/dataloader/.venv/bin/python3
cachedir: .pytest_cache
benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /data/OpenSock/dataloader
configfile: pytest.ini
plugins: benchmark-5.2.3
collecting ... collected 108 items

tests/test_augmentations.py::TestBasics::test_augment_raster_executes_successfully SKIPPED [  0%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_with_incorrect_shape SKIPPED [  1%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_with_mismatched_dtype SKIPPED [  2%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=uint8] SKIPPED [  3%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=int32] SKIPPED [  4%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=float32] SKIPPED [  5%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_for_unsupported_dtype SKIPPED [  6%]
tests/test_augmentations.py::TestBasics::test_pipe_fails_with_no_augmentation SKIPPED [  7%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip] SKIPPED [  8%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip_2_swap] SKIPPED [  9%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_swap_2_skip] SKIPPED [ 10%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip_2_skip] SKIPPED [ 11%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_swap_2_skip_3_swap] SKIPPED [ 12%]
tests/test_augmentations.py::TestAugmentationRules::test_random_crop_fails_with_different_min_max SKIPPED [ 12%]
tests/test_augmentations.py::TestAugmentationRules::test_random_crop_succeedes_with_same_min_max SKIPPED [ 13%]
tests/test_augmentations.py::TestAugmentationRules::test_flip_fails_if_disabled SKIPPED [ 14%]
tests/test_augmentations.py::TestAugmentationRules::test_flip_tries_different_settings SKIPPED [ 15%]
tests/test_augmentations.py::TestAugmentationRules::test_resize_fails_if_not_three_channels SKIPPED [ 16%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=uint8] SKIPPED [ 17%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=int32] SKIPPED [ 18%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=float32] SKIPPED [ 19%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=uint8] SKIPPED [ 20%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=int32] SKIPPED [ 21%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=float32] SKIPPED [ 22%]
tests/test_augmentations.py::TestRasterCorrectness::test_resize[dtype=uint8] SKIPPED [ 23%]
tests/test_augmentations.py::TestRasterCorrectness::test_resize[dtype=float32] SKIPPED [ 24%]
tests/test_augmentations.py::TestPointCorrectness::test_flip[dtype=int32] SKIPPED [ 25%]
tests/test_augmentations.py::TestPointCorrectness::test_flip[dtype=float32] SKIPPED [ 25%]
tests/test_augmentations.py::TestPointCorrectness::test_random_crop[dtype=int32] SKIPPED [ 26%]
tests/test_augmentations.py::TestPointCorrectness::test_random_crop[dtype=float32] SKIPPED [ 27%]
tests/test_augmentations.py::TestPointCorrectness::test_resize[dtype=int32] SKIPPED [ 28%]
tests/test_augmentations.py::TestPointCorrectness::test_resize[dtype=float32] SKIPPED [ 29%]
tests/test_bindings.py::test_binding[jax] SKIPPED (Skipping non-benc...) [ 30%]
tests/test_bindings.py::test_binding[pytorch] SKIPPED (Skipping non-...) [ 31%]
tests/test_compression.py::test_prepare_dense_features_jpg_only SKIPPED  [ 32%]
tests/test_compression.py::test_compress_many_files[fp16] SKIPPED (S...) [ 33%]
tests/test_compression.py::test_compress_many_files[fp32] SKIPPED (S...) [ 34%]
tests/test_compression.py::test_compress_many_files[fp16_permute] SKIPPED [ 35%]
tests/test_compression.py::test_compress_many_files[fp32_permute] SKIPPED [ 36%]
tests/test_compression.py::test_compress_many_files[fp16_permute_bitshuffle] SKIPPED [ 37%]
tests/test_compression.py::test_compress_many_files[fp16_permute_bitshuffle_compress] SKIPPED [ 37%]
tests/test_compression.py::TestBenchmarks::test_compress[fp32_zstd7] PASSED [ 38%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_zstd7] PASSED [ 39%]
tests/test_compression.py::TestBenchmarks::test_compress[fp32_bshuf_zstd7] PASSED [ 40%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd7] PASSED [ 41%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd(3,7,22)] PASSED [ 42%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd22] PASSED [ 43%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp32_zstd7] PASSED [ 44%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_zstd7] PASSED [ 45%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp32_bshuf_zstd7] PASSED [ 46%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd7] PASSED [ 47%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd(3,7,22)] PASSED [ 48%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd22] PASSED [ 49%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=16,prefetch=16] SKIPPED [ 50%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=16,prefetch=4] SKIPPED [ 50%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=16] SKIPPED [ 51%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=1] SKIPPED [ 52%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=2] SKIPPED [ 53%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=16,prefetch=16] SKIPPED [ 54%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=16,prefetch=4] SKIPPED [ 55%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=16] SKIPPED [ 56%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=1] SKIPPED [ 57%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=2] SKIPPED [ 58%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=16,prefetch=16] SKIPPED [ 59%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=16,prefetch=4] SKIPPED [ 60%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=16] SKIPPED [ 61%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=1] SKIPPED [ 62%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=2] SKIPPED [ 62%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=16,prefetch=16] SKIPPED [ 63%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=16,prefetch=4] SKIPPED [ 64%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=16] SKIPPED [ 65%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=1] SKIPPED [ 66%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=2] SKIPPED [ 67%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=16,prefetch=16] SKIPPED [ 68%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=16,prefetch=4] SKIPPED [ 69%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=16] SKIPPED [ 70%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=1] SKIPPED [ 71%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=2] SKIPPED [ 72%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=16] SKIPPED [ 73%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=4] SKIPPED [ 74%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=16] SKIPPED [ 75%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=1] SKIPPED [ 75%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=2] SKIPPED [ 76%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=16,prefetch=16] SKIPPED [ 77%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=16,prefetch=4] SKIPPED [ 78%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=16] SKIPPED [ 79%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=1] SKIPPED [ 80%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=2] SKIPPED [ 81%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=16,prefetch=16] SKIPPED [ 82%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=16,prefetch=4] SKIPPED [ 83%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=16] SKIPPED [ 84%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=1] SKIPPED [ 85%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=2] SKIPPED [ 86%]
tests/test_dataloader.py::TestRaster::test_aug_pipe_buffers_upsize SKIPPED [ 87%]
tests/test_dataloader.py::TestDecoders::test_exr SKIPPED (Skipping n...) [ 87%]
tests/test_dataloader.py::TestDecoders::test_png SKIPPED (Skipping n...) [ 88%]
tests/test_dataloader.py::TestDecoders::test_npy[float32] SKIPPED (S...) [ 89%]
tests/test_dataloader.py::TestDecoders::test_compressed SKIPPED (Ski...) [ 90%]
tests/test_dataloader.py::TestPoints::test_points_must_follow_raster_item SKIPPED [ 91%]
tests/test_dataloader.py::TestPoints::test_points_batch_returns_lengths_metadata SKIPPED [ 92%]
tests/test_dataloader.py::TestPoints::test_points_tensor_prefix_matches_lengths SKIPPED [ 93%]
tests/test_dataloader.py::TestPoints::test_points_metadata_matches_dataset_iteration_order SKIPPED [ 94%]
tests/test_dataloader.py::test_end_to_end_perf PASSED                    [ 95%]
tests/test_dataset.py::test_get_eroneous_dataset SKIPPED (Skipping n...) [ 96%]
tests/test_dataset.py::test_get_dataset SKIPPED (Skipping non-benchm...) [ 97%]
tests/test_dataset.py::test_dataset_works_with_trailing_slash SKIPPED    [ 98%]
tests/test_dataset.py::test_split_dataset SKIPPED (Skipping non-benc...) [ 99%]
tests/test_meta.py::test_version SKIPPED (Skipping non-benchmark (--...) [100%]
Wrote benchmark data in: <_io.BufferedWriter name='.benchmark.json'>



---------------------------------------------------------------------------------- benchmark 'Compression': 6 tests ---------------------------------------------------------------------------------
Name (time in s)                              Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_compress[fp16_bshuf_zstd7]            2.1714 (1.0)      2.2026 (1.0)      2.1874 (1.0)      0.0156 (1.0)      2.1883 (1.0)      0.0234 (1.0)           1;0  0.4572 (1.0)           3           1
test_compress[fp16_zstd7]                  2.3033 (1.06)     2.3737 (1.08)     2.3307 (1.07)     0.0377 (2.41)     2.3151 (1.06)     0.0527 (2.26)          1;0  0.4291 (0.94)          3           1
test_compress[fp32_bshuf_zstd7]            3.0437 (1.40)     3.1611 (1.44)     3.0904 (1.41)     0.0622 (3.99)     3.0666 (1.40)     0.0880 (3.76)          1;0  0.3236 (0.71)          3           1
test_compress[fp32_zstd7]                  3.5379 (1.63)     3.8113 (1.73)     3.6584 (1.67)     0.1396 (8.94)     3.6259 (1.66)     0.2050 (8.77)          1;0  0.2733 (0.60)          3           1
test_compress[fp16_bshuf_zstd22]           8.0080 (3.69)     8.0585 (3.66)     8.0332 (3.67)     0.0252 (1.62)     8.0332 (3.67)     0.0378 (1.62)          1;0  0.1245 (0.27)          3           1
test_compress[fp16_bshuf_zstd(3,7,22)]     8.4693 (3.90)     8.5209 (3.87)     8.4868 (3.88)     0.0296 (1.90)     8.4701 (3.87)     0.0387 (1.66)          1;0  0.1178 (0.26)          3           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------ benchmark 'Dataloader': 1 tests ------------------------------------------
Name (time in ms)            Min       Max     Mean   StdDev   Median      IQR  Outliers      OPS  Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------
test_end_to_end_perf     36.1742  106.1146  74.5050  35.4513  81.2261  52.4553       1;0  13.4219       3           1
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------- benchmark 'Decompression': 6 tests ---------------------------------------------------------------------------------
Name (time in s)                                Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_decompress[fp16_bshuf_zstd7]            2.6098 (1.0)      2.6135 (1.0)      2.6121 (1.0)      0.0020 (1.0)      2.6129 (1.0)      0.0028 (1.0)           1;0  0.3828 (1.0)           3           1
test_decompress[fp16_bshuf_zstd(3,7,22)]     2.6441 (1.01)     2.6609 (1.02)     2.6548 (1.02)     0.0093 (4.59)     2.6594 (1.02)     0.0126 (4.44)          1;0  0.3767 (0.98)          3           1
test_decompress[fp16_bshuf_zstd22]           2.6492 (1.02)     2.6702 (1.02)     2.6565 (1.02)     0.0119 (5.88)     2.6502 (1.01)     0.0158 (5.57)          1;0  0.3764 (0.98)          3           1
test_decompress[fp16_zstd7]                  2.6495 (1.02)     2.6657 (1.02)     2.6591 (1.02)     0.0086 (4.23)     2.6622 (1.02)     0.0122 (4.30)          1;0  0.3761 (0.98)          3           1
test_decompress[fp32_bshuf_zstd7]            2.8792 (1.10)     2.8843 (1.10)     2.8810 (1.10)     0.0029 (1.43)     2.8795 (1.10)     0.0039 (1.37)          1;0  0.3471 (0.91)          3           1
test_decompress[fp32_zstd7]                  2.8853 (1.11)     2.8973 (1.11)     2.8896 (1.11)     0.0066 (3.28)     2.8864 (1.10)     0.0090 (3.17)          1;0  0.3461 (0.90)          3           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
================== 13 passed, 95 skipped in 220.34s (0:03:40) ==================
```
