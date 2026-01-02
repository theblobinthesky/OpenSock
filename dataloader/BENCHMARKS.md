# Benchmark Results

Generated: 2026-01-02 15:12:14

Command: `pytest ./tests --benchmark-only`

## System

- platform: Linux-6.8.0-90-generic-x86_64-with-glibc2.39
- python: 3.12.3
- cpu: 13th Gen Intel(R) Core(TM) i7-13700K
- cpu_count: 24
- mem_total: 32597212 kB
- gpu: NVIDIA GeForce RTX 4080, 565.57.01, 16376 MiB

## Compression / test_compress[fp32_zstd7]

- mean: 3.484791464000106
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 201.77184768838111
  - c_out_MBps: 176.52996207535764
  - ratio: 0.874898872651415

## Compression / test_compress[fp16_zstd7]

- mean: 2.297311833334485
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 306.0676405777365
  - c_out_MBps: 119.87836870860694
  - ratio: 0.39167279651753867

## Compression / test_compress[fp32_bshuf_zstd7]

- mean: 2.9464503156668798
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 238.63725404134556
  - c_out_MBps: 175.32681557595208
  - ratio: 0.7347001048946679

## Compression / test_compress[fp16_bshuf_zstd7]

- mean: 2.1624231719991562
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 325.1596734648173
  - c_out_MBps: 110.2526511278687
  - ratio: 0.33907233930039665

## Compression / test_compress[fp16_bshuf_zstd(3,7,22)]

- mean: 8.495166000666359
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 82.76857832381927
  - c_out_MBps: 27.871339088972174
  - ratio: 0.3367381638467906

## Compression / test_compress[fp16_bshuf_zstd22]

- mean: 7.9759104066664195
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 88.1570600281954
  - c_out_MBps: 29.58637248838086
  - ratio: 0.33560979096760035

## Decompression / test_decompress[fp32_zstd7]

- mean: 2.8746704359994815
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 244.59597305300525
  - d_out_MBps: 244.59325535016802

## Decompression / test_decompress[fp16_zstd7]

- mean: 2.6292950176660574
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 267.42256299719037
  - d_out_MBps: 133.70979583419702

## Decompression / test_decompress[fp32_bshuf_zstd7]

- mean: 2.880752513333088
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 244.0795622830027
  - d_out_MBps: 244.07685031799917

## Decompression / test_decompress[fp16_bshuf_zstd7]

- mean: 2.6001482920000853
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 270.4202735910637
  - d_out_MBps: 135.20863447737096

## Decompression / test_decompress[fp16_bshuf_zstd(3,7,22)]

- mean: 2.6290787943338123
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 267.4445566315437
  - d_out_MBps: 133.7207925291882

## Decompression / test_decompress[fp16_bshuf_zstd22]

- mean: 2.6475656460000514
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 265.57710233258797
  - d_out_MBps: 132.78707575434115

## Dataloader / test_end_to_end_perf[threads=16,prefetch=16]

- mean: 0.6058821300008276
- rounds: 10
- iterations: 1
- extra:
  - batches: 32
  - in_MBps: 43.391300075573355
  - items: 691200000
  - out_MBps: 870.3734998741087

## Dataloader / test_end_to_end_perf[threads=16,prefetch=4]

- mean: 0.6192557538004622
- rounds: 10
- iterations: 1
- extra:
  - batches: 32
  - in_MBps: 42.454209188930164
  - items: 691200000
  - out_MBps: 851.5766656403515

## Dataloader / test_end_to_end_perf[threads=8,prefetch=16]

- mean: 0.4596152552003332
- rounds: 10
- iterations: 1
- extra:
  - batches: 32
  - in_MBps: 57.20004507212101
  - items: 691200000
  - out_MBps: 1147.359109675648

## Dataloader / test_end_to_end_perf[threads=8,prefetch=1]

- mean: 0.4645837037001911
- rounds: 10
- iterations: 1
- extra:
  - batches: 32
  - in_MBps: 56.58832435125434
  - items: 691200000
  - out_MBps: 1135.0887812033754

## Dataloader / test_end_to_end_perf[threads=8,prefetch=2]

- mean: 0.45598424470044846
- rounds: 10
- iterations: 1
- extra:
  - batches: 32
  - in_MBps: 57.65553002947341
  - items: 691200000
  - out_MBps: 1156.4955502934756

## Raw Output

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0 -- /data/OpenSock/dataloader/.venv/bin/python3
cachedir: .pytest_cache
benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /data/OpenSock/dataloader
configfile: pytest.ini
plugins: benchmark-5.2.3
collecting ... collected 112 items

tests/test_augmentations.py::TestBasics::test_augment_raster_executes_successfully SKIPPED [  0%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_with_incorrect_shape SKIPPED [  1%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_with_mismatched_dtype SKIPPED [  2%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=uint8] SKIPPED [  3%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=int32] SKIPPED [  4%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=float32] SKIPPED [  5%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_for_unsupported_dtype SKIPPED [  6%]
tests/test_augmentations.py::TestBasics::test_pipe_fails_with_no_augmentation SKIPPED [  7%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip] SKIPPED [  8%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip_2_swap] SKIPPED [  8%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_swap_2_skip] SKIPPED [  9%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip_2_skip] SKIPPED [ 10%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_swap_2_skip_3_swap] SKIPPED [ 11%]
tests/test_augmentations.py::TestAugmentationRules::test_random_crop_fails_with_different_min_max SKIPPED [ 12%]
tests/test_augmentations.py::TestAugmentationRules::test_random_crop_succeedes_with_same_min_max SKIPPED [ 13%]
tests/test_augmentations.py::TestAugmentationRules::test_flip_fails_if_disabled SKIPPED [ 14%]
tests/test_augmentations.py::TestAugmentationRules::test_flip_tries_different_settings SKIPPED [ 15%]
tests/test_augmentations.py::TestAugmentationRules::test_resize_fails_if_not_three_channels SKIPPED [ 16%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=uint8] SKIPPED [ 16%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=int32] SKIPPED [ 17%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=float32] SKIPPED [ 18%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=uint8] SKIPPED [ 19%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=int32] SKIPPED [ 20%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=float32] SKIPPED [ 21%]
tests/test_augmentations.py::TestRasterCorrectness::test_resize[dtype=uint8] SKIPPED [ 22%]
tests/test_augmentations.py::TestRasterCorrectness::test_resize[dtype=float32] SKIPPED [ 23%]
tests/test_augmentations.py::TestPointCorrectness::test_flip[dtype=int32] SKIPPED [ 24%]
tests/test_augmentations.py::TestPointCorrectness::test_flip[dtype=float32] SKIPPED [ 25%]
tests/test_augmentations.py::TestPointCorrectness::test_random_crop[dtype=int32] SKIPPED [ 25%]
tests/test_augmentations.py::TestPointCorrectness::test_random_crop[dtype=float32] SKIPPED [ 26%]
tests/test_augmentations.py::TestPointCorrectness::test_resize[dtype=int32] SKIPPED [ 27%]
tests/test_augmentations.py::TestPointCorrectness::test_resize[dtype=float32] SKIPPED [ 28%]
tests/test_bindings.py::test_binding[jax] SKIPPED (Skipping non-benc...) [ 29%]
tests/test_bindings.py::test_binding[pytorch] SKIPPED (Skipping non-...) [ 30%]
tests/test_compression.py::test_prepare_dense_features_jpg_only SKIPPED  [ 31%]
tests/test_compression.py::test_compress_many_files[fp16] SKIPPED (S...) [ 32%]
tests/test_compression.py::test_compress_many_files[fp32] SKIPPED (S...) [ 33%]
tests/test_compression.py::test_compress_many_files[fp16_permute] SKIPPED [ 33%]
tests/test_compression.py::test_compress_many_files[fp32_permute] SKIPPED [ 34%]
tests/test_compression.py::test_compress_many_files[fp16_permute_bitshuffle] SKIPPED [ 35%]
tests/test_compression.py::test_compress_many_files[fp16_permute_bitshuffle_compress] SKIPPED [ 36%]
tests/test_compression.py::TestBenchmarks::test_compress[fp32_zstd7] PASSED [ 37%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_zstd7] PASSED [ 38%]
tests/test_compression.py::TestBenchmarks::test_compress[fp32_bshuf_zstd7] PASSED [ 39%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd7] PASSED [ 40%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd(3,7,22)] PASSED [ 41%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd22] PASSED [ 41%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp32_zstd7] PASSED [ 42%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_zstd7] PASSED [ 43%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp32_bshuf_zstd7] PASSED [ 44%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd7] PASSED [ 45%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd(3,7,22)] PASSED [ 46%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd22] PASSED [ 47%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=16,prefetch=16] SKIPPED [ 48%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=16,prefetch=4] SKIPPED [ 49%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=16] SKIPPED [ 50%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=1] SKIPPED [ 50%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=2] SKIPPED [ 51%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=16,prefetch=16] SKIPPED [ 52%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=16,prefetch=4] SKIPPED [ 53%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=16] SKIPPED [ 54%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=1] SKIPPED [ 55%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=2] SKIPPED [ 56%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=16,prefetch=16] SKIPPED [ 57%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=16,prefetch=4] SKIPPED [ 58%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=16] SKIPPED [ 58%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=1] SKIPPED [ 59%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=2] SKIPPED [ 60%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=16,prefetch=16] SKIPPED [ 61%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=16,prefetch=4] SKIPPED [ 62%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=16] SKIPPED [ 63%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=1] SKIPPED [ 64%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=2] SKIPPED [ 65%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=16,prefetch=16] SKIPPED [ 66%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=16,prefetch=4] SKIPPED [ 66%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=16] SKIPPED [ 67%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=1] SKIPPED [ 68%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=2] SKIPPED [ 69%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=16] SKIPPED [ 70%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=4] SKIPPED [ 71%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=16] SKIPPED [ 72%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=1] SKIPPED [ 73%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=2] SKIPPED [ 74%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=16,prefetch=16] SKIPPED [ 75%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=16,prefetch=4] SKIPPED [ 75%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=16] SKIPPED [ 76%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=1] SKIPPED [ 77%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=2] SKIPPED [ 78%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=16,prefetch=16] SKIPPED [ 79%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=16,prefetch=4] SKIPPED [ 80%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=16] SKIPPED [ 81%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=1] SKIPPED [ 82%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=2] SKIPPED [ 83%]
tests/test_dataloader.py::TestRaster::test_aug_pipe_buffers_upsize SKIPPED [ 83%]
tests/test_dataloader.py::TestDecoders::test_exr SKIPPED (Skipping n...) [ 84%]
tests/test_dataloader.py::TestDecoders::test_png SKIPPED (Skipping n...) [ 85%]
tests/test_dataloader.py::TestDecoders::test_npy[float32] SKIPPED (S...) [ 86%]
tests/test_dataloader.py::TestDecoders::test_compressed SKIPPED (Ski...) [ 87%]
tests/test_dataloader.py::TestPoints::test_points_must_follow_raster_item SKIPPED [ 88%]
tests/test_dataloader.py::TestPoints::test_points_batch_returns_lengths_metadata SKIPPED [ 89%]
tests/test_dataloader.py::TestPoints::test_points_tensor_prefix_matches_lengths SKIPPED [ 90%]
tests/test_dataloader.py::TestPoints::test_points_metadata_matches_dataset_iteration_order SKIPPED [ 91%]
tests/test_dataloader.py::test_end_to_end_perf[threads=16,prefetch=16] PASSED [ 91%]
tests/test_dataloader.py::test_end_to_end_perf[threads=16,prefetch=4] PASSED [ 92%]
tests/test_dataloader.py::test_end_to_end_perf[threads=8,prefetch=16] PASSED [ 93%]
tests/test_dataloader.py::test_end_to_end_perf[threads=8,prefetch=1] PASSED [ 94%]
tests/test_dataloader.py::test_end_to_end_perf[threads=8,prefetch=2] PASSED [ 95%]
tests/test_dataset.py::test_get_eroneous_dataset SKIPPED (Skipping n...) [ 96%]
tests/test_dataset.py::test_get_dataset SKIPPED (Skipping non-benchm...) [ 97%]
tests/test_dataset.py::test_dataset_works_with_trailing_slash SKIPPED    [ 98%]
tests/test_dataset.py::test_split_dataset SKIPPED (Skipping non-benc...) [ 99%]
tests/test_meta.py::test_version SKIPPED (Skipping non-benchmark (--...) [100%]
Wrote benchmark data in: <_io.BufferedWriter name='.benchmark.json'>



---------------------------------------------------------------------------------- benchmark 'Compression': 6 tests ---------------------------------------------------------------------------------
Name (time in s)                              Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_compress[fp16_bshuf_zstd7]            2.1371 (1.0)      2.1989 (1.0)      2.1624 (1.0)      0.0324 (1.10)     2.1513 (1.0)      0.0464 (1.11)          1;0  0.4624 (1.0)           3           1
test_compress[fp16_zstd7]                  2.2394 (1.05)     2.3343 (1.06)     2.2973 (1.06)     0.0508 (1.72)     2.3182 (1.08)     0.0711 (1.70)          1;0  0.4353 (0.94)          3           1
test_compress[fp32_bshuf_zstd7]            2.9241 (1.37)     2.9798 (1.36)     2.9465 (1.36)     0.0295 (1.0)      2.9354 (1.36)     0.0418 (1.0)           1;0  0.3394 (0.73)          3           1
test_compress[fp32_zstd7]                  3.4279 (1.60)     3.5196 (1.60)     3.4848 (1.61)     0.0497 (1.69)     3.5069 (1.63)     0.0688 (1.65)          1;0  0.2870 (0.62)          3           1
test_compress[fp16_bshuf_zstd22]           7.8939 (3.69)     8.0361 (3.65)     7.9759 (3.69)     0.0736 (2.50)     7.9978 (3.72)     0.1066 (2.55)          1;0  0.1254 (0.27)          3           1
test_compress[fp16_bshuf_zstd(3,7,22)]     8.4071 (3.93)     8.5659 (3.90)     8.4952 (3.93)     0.0808 (2.74)     8.5125 (3.96)     0.1191 (2.85)          1;0  0.1177 (0.25)          3           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------ benchmark 'Dataloader': 5 tests ------------------------------------------------------------------------------------------
Name (time in ms)                                     Min                 Max                Mean             StdDev              Median                IQR            Outliers     OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_end_to_end_perf[threads=8,prefetch=2]       433.1626 (1.0)      495.6317 (1.04)     455.9842 (1.0)      16.2627 (1.92)     453.6015 (1.0)       7.8368 (1.0)           2;2  2.1931 (1.0)          10           1
test_end_to_end_perf[threads=8,prefetch=16]      447.2309 (1.03)     475.0851 (1.0)      459.6153 (1.01)      8.4507 (1.0)      460.1806 (1.01)     12.2686 (1.57)          3;0  2.1757 (0.99)         10           1
test_end_to_end_perf[threads=8,prefetch=1]       449.9359 (1.04)     485.3143 (1.02)     464.5837 (1.02)     11.2842 (1.34)     464.5029 (1.02)     17.4804 (2.23)          3;0  2.1525 (0.98)         10           1
test_end_to_end_perf[threads=16,prefetch=16]     583.9743 (1.35)     636.0607 (1.34)     605.8821 (1.33)     21.0125 (2.49)     600.9712 (1.32)     46.2181 (5.90)          5;0  1.6505 (0.75)         10           1
test_end_to_end_perf[threads=16,prefetch=4]      585.7490 (1.35)     640.5614 (1.35)     619.2558 (1.36)     19.2333 (2.28)     623.5016 (1.37)     27.8515 (3.55)          4;0  1.6148 (0.74)         10           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------- benchmark 'Decompression': 6 tests ---------------------------------------------------------------------------------
Name (time in s)                                Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_decompress[fp16_bshuf_zstd7]            2.5992 (1.0)      2.6018 (1.0)      2.6001 (1.0)      0.0014 (1.0)      2.5994 (1.0)      0.0019 (1.0)           1;0  0.3846 (1.0)           3           1
test_decompress[fp16_bshuf_zstd(3,7,22)]     2.6266 (1.01)     2.6304 (1.01)     2.6291 (1.01)     0.0022 (1.49)     2.6302 (1.01)     0.0029 (1.48)          1;0  0.3804 (0.99)          3           1
test_decompress[fp16_zstd7]                  2.6273 (1.01)     2.6326 (1.01)     2.6293 (1.01)     0.0029 (1.99)     2.6280 (1.01)     0.0040 (2.04)          1;0  0.3803 (0.99)          3           1
test_decompress[fp16_bshuf_zstd22]           2.6456 (1.02)     2.6488 (1.02)     2.6476 (1.02)     0.0017 (1.20)     2.6483 (1.02)     0.0024 (1.25)          1;0  0.3777 (0.98)          3           1
test_decompress[fp32_zstd7]                  2.8667 (1.10)     2.8791 (1.11)     2.8747 (1.11)     0.0069 (4.79)     2.8782 (1.11)     0.0093 (4.79)          1;0  0.3479 (0.90)          3           1
test_decompress[fp32_bshuf_zstd7]            2.8746 (1.11)     2.8865 (1.11)     2.8808 (1.11)     0.0060 (4.13)     2.8812 (1.11)     0.0089 (4.60)          1;0  0.3471 (0.90)          3           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
================== 17 passed, 95 skipped in 254.37s (0:04:14) ==================
```
