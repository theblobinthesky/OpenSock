# Benchmark Results

Generated: 2026-01-02 15:53:05

Command: `pytest ./tests --benchmark-only`

## System

- platform: Linux-6.8.0-90-generic-x86_64-with-glibc2.39
- python: 3.12.3
- cpu: 13th Gen Intel(R) Core(TM) i7-13700K
- cpu_count: 24
- mem_total: 32597212 kB
- gpu: NVIDIA GeForce RTX 4080, 565.57.01, 16376 MiB

## Compression / test_compress[fp32_zstd7]

- mean: 3.4878700519996833
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 201.59375263905727
  - c_out_MBps: 176.37414691747944
  - ratio: 0.874898872651415

## Compression / test_compress[fp16_zstd7]

- mean: 2.317812646000675
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 303.36050401366015
  - c_out_MBps: 118.81805696000028
  - ratio: 0.39167279651753867

## Compression / test_compress[fp32_bshuf_zstd7]

- mean: 2.950341056332642
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 238.32255290987075
  - c_out_MBps: 175.09560462164706
  - ratio: 0.7347001048946679

## Compression / test_compress[fp16_bshuf_zstd7]

- mean: 2.145169525665551
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 327.7749399697672
  - c_out_MBps: 111.13941565959605
  - ratio: 0.33907233930039665

## Compression / test_compress[fp16_bshuf_zstd(3,7,22)]

- mean: 8.475670023001536
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 82.95896496581585
  - c_out_MBps: 27.935449537219057
  - ratio: 0.3367381638467906

## Compression / test_compress[fp16_bshuf_zstd22]

- mean: 7.986903228667264
- rounds: 3
- iterations: 1
- extra:
  - c_in_MBps: 88.03572453166286
  - c_out_MBps: 29.545651107752622
  - ratio: 0.33560979096760035

## Decompression / test_decompress[fp32_zstd7]

- mean: 2.8718071616676752
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 244.83984227258722
  - d_out_MBps: 244.8371218601221

## Decompression / test_decompress[fp16_zstd7]

- mean: 2.632778518665873
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 267.0687289169708
  - d_out_MBps: 133.5328807598103

## Decompression / test_decompress[fp32_bshuf_zstd7]

- mean: 2.879591384333859
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 244.17798175301076
  - d_out_MBps: 244.1752686944697

## Decompression / test_decompress[fp16_bshuf_zstd7]

- mean: 2.6043387323334173
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 269.98516121211776
  - d_out_MBps: 134.99108070516215

## Decompression / test_decompress[fp16_bshuf_zstd(3,7,22)]

- mean: 2.6332231853327053
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 267.02362960212196
  - d_out_MBps: 133.51033135293483

## Decompression / test_decompress[fp16_bshuf_zstd22]

- mean: 2.6444214486667383
- rounds: 3
- iterations: 1
- extra:
  - d_in_MBps: 265.89287152186154
  - d_out_MBps: 132.94495859472417

## Dataloader / test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=16]

- mean: 0.8857360230998893
- rounds: 10
- iterations: 1
- extra:
  - batches: 400
  - in_MBps: 373.60666165763143
  - items: 1728000000
  - out_MBps: 7442.168663221014

## Dataloader / test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=4]

- mean: 0.8696026161000191
- rounds: 10
- iterations: 1
- extra:
  - batches: 400
  - in_MBps: 380.53804412910745
  - items: 1728000000
  - out_MBps: 7580.240391367258

## Dataloader / test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=2]

- mean: 0.882227072299429
- rounds: 10
- iterations: 1
- extra:
  - batches: 400
  - in_MBps: 375.0926366811182
  - items: 1728000000
  - out_MBps: 7471.768983260962

## Dataloader / test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=1]

- mean: 0.8901845666994632
- rounds: 10
- iterations: 1
- extra:
  - batches: 400
  - in_MBps: 371.7396269036619
  - items: 1728000000
  - out_MBps: 7404.977710903708

## Dataloader / test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=16]

- mean: 0.9342066814991995
- rounds: 10
- iterations: 1
- extra:
  - batches: 400
  - in_MBps: 354.22234207232
  - items: 1728000000
  - out_MBps: 7056.03696220797

## Dataloader / test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=4]

- mean: 0.9513504993003152
- rounds: 10
- iterations: 1
- extra:
  - batches: 400
  - in_MBps: 347.83907607515215
  - items: 1728000000
  - out_MBps: 6928.883602676442

## Dataloader / test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=2]

- mean: 0.9596467905001191
- rounds: 10
- iterations: 1
- extra:
  - batches: 400
  - in_MBps: 344.8319548151662
  - items: 1728000000
  - out_MBps: 6868.982359191438

## Dataloader / test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=1]

- mean: 0.9406244210000295
- rounds: 10
- iterations: 1
- extra:
  - batches: 400
  - in_MBps: 351.80553610168914
  - items: 1728000000
  - out_MBps: 7007.89467914505

## Dataloader / test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=16]

- mean: 0.9110844843995437
- rounds: 10
- iterations: 1
- extra:
  - batches: 200
  - in_MBps: 363.21206690107266
  - items: 1728000000
  - out_MBps: 7235.11045119418

## Dataloader / test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=4]

- mean: 0.8977483839997149
- rounds: 10
- iterations: 1
- extra:
  - batches: 200
  - in_MBps: 368.60760163770055
  - items: 1728000000
  - out_MBps: 7342.588405040329

## Dataloader / test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=2]

- mean: 0.8917552937004075
- rounds: 10
- iterations: 1
- extra:
  - batches: 200
  - in_MBps: 371.0848492158552
  - items: 1728000000
  - out_MBps: 7391.934672623955

## Dataloader / test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=1]

- mean: 0.8926462193998305
- rounds: 10
- iterations: 1
- extra:
  - batches: 200
  - in_MBps: 370.71447960956795
  - items: 1728000000
  - out_MBps: 7384.556985444901

## Dataloader / test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=16]

- mean: 1.0430542482004967
- rounds: 10
- iterations: 1
- extra:
  - batches: 200
  - in_MBps: 317.2575916076872
  - items: 1728000000
  - out_MBps: 6319.706656075015

## Dataloader / test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=4]

- mean: 1.0444552443997963
- rounds: 10
- iterations: 1
- extra:
  - batches: 200
  - in_MBps: 316.8320332293607
  - items: 1728000000
  - out_MBps: 6311.229619789045

## Dataloader / test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=2]

- mean: 1.0130143558999407
- rounds: 10
- iterations: 1
- extra:
  - batches: 200
  - in_MBps: 326.66553713967534
  - items: 1728000000
  - out_MBps: 6507.11101635276

## Dataloader / test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=1]

- mean: 1.0307174866000424
- rounds: 10
- iterations: 1
- extra:
  - batches: 200
  - in_MBps: 321.05487973414455
  - items: 1728000000
  - out_MBps: 6395.347862724161

## Dataloader / test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=16]

- mean: 0.9480796519994328
- rounds: 10
- iterations: 1
- extra:
  - batches: 100
  - in_MBps: 349.03911079873524
  - items: 1728000000
  - out_MBps: 6952.788050137315

## Dataloader / test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=4]

- mean: 0.9563095495002927
- rounds: 10
- iterations: 1
- extra:
  - batches: 100
  - in_MBps: 346.03531761569536
  - items: 1728000000
  - out_MBps: 6892.953101267743

## Dataloader / test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=2]

- mean: 0.9386038150998501
- rounds: 10
- iterations: 1
- extra:
  - batches: 100
  - in_MBps: 352.56289541615905
  - items: 1728000000
  - out_MBps: 7022.9811225503645

## Dataloader / test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=1]

- mean: 0.9494047359010438
- rounds: 10
- iterations: 1
- extra:
  - batches: 100
  - in_MBps: 348.5519570177789
  - items: 1728000000
  - out_MBps: 6943.084045966947

## Dataloader / test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=16]

- mean: 1.0549526863003849
- rounds: 10
- iterations: 1
- extra:
  - batches: 100
  - in_MBps: 313.6793554796749
  - items: 1728000000
  - out_MBps: 6248.4289206530975

## Dataloader / test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=4]

- mean: 1.0910991368004033
- rounds: 10
- iterations: 1
- extra:
  - batches: 100
  - in_MBps: 303.28763678674926
  - items: 1728000000
  - out_MBps: 6041.427999228496

## Dataloader / test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=2]

- mean: 1.0887924609996844
- rounds: 10
- iterations: 1
- extra:
  - batches: 100
  - in_MBps: 303.9301708577428
  - items: 1728000000
  - out_MBps: 6054.227147153171

## Dataloader / test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=1]

- mean: 1.0669163652004499
- rounds: 10
- iterations: 1
- extra:
  - batches: 100
  - in_MBps: 310.161967229816
  - items: 1728000000
  - out_MBps: 6178.363262580144

## Raw Output

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0 -- /data/OpenSock/dataloader/.venv/bin/python3
cachedir: .pytest_cache
benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /data/OpenSock/dataloader
configfile: pytest.ini
plugins: benchmark-5.2.3
collecting ... collected 131 items

tests/test_augmentations.py::TestBasics::test_augment_raster_executes_successfully SKIPPED [  0%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_with_incorrect_shape SKIPPED [  1%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_with_mismatched_dtype SKIPPED [  2%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=uint8] SKIPPED [  3%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=int32] SKIPPED [  3%]
tests/test_augmentations.py::TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=float32] SKIPPED [  4%]
tests/test_augmentations.py::TestBasics::test_augment_raster_fails_for_unsupported_dtype SKIPPED [  5%]
tests/test_augmentations.py::TestBasics::test_pipe_fails_with_no_augmentation SKIPPED [  6%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip] SKIPPED [  6%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip_2_swap] SKIPPED [  7%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_swap_2_skip] SKIPPED [  8%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_skip_2_skip] SKIPPED [  9%]
tests/test_augmentations.py::TestBasics::test_augmentations_are_skipped[1_swap_2_skip_3_swap] SKIPPED [  9%]
tests/test_augmentations.py::TestAugmentationRules::test_random_crop_fails_with_different_min_max SKIPPED [ 10%]
tests/test_augmentations.py::TestAugmentationRules::test_random_crop_succeedes_with_same_min_max SKIPPED [ 11%]
tests/test_augmentations.py::TestAugmentationRules::test_flip_fails_if_disabled SKIPPED [ 12%]
tests/test_augmentations.py::TestAugmentationRules::test_flip_tries_different_settings SKIPPED [ 12%]
tests/test_augmentations.py::TestAugmentationRules::test_resize_fails_if_not_three_channels SKIPPED [ 13%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=uint8] SKIPPED [ 14%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=int32] SKIPPED [ 15%]
tests/test_augmentations.py::TestRasterCorrectness::test_flip[dtype=float32] SKIPPED [ 16%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=uint8] SKIPPED [ 16%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=int32] SKIPPED [ 17%]
tests/test_augmentations.py::TestRasterCorrectness::test_random_crop[dtype=float32] SKIPPED [ 18%]
tests/test_augmentations.py::TestRasterCorrectness::test_resize[dtype=uint8] SKIPPED [ 19%]
tests/test_augmentations.py::TestRasterCorrectness::test_resize[dtype=float32] SKIPPED [ 19%]
tests/test_augmentations.py::TestPointCorrectness::test_flip[dtype=int32] SKIPPED [ 20%]
tests/test_augmentations.py::TestPointCorrectness::test_flip[dtype=float32] SKIPPED [ 21%]
tests/test_augmentations.py::TestPointCorrectness::test_random_crop[dtype=int32] SKIPPED [ 22%]
tests/test_augmentations.py::TestPointCorrectness::test_random_crop[dtype=float32] SKIPPED [ 22%]
tests/test_augmentations.py::TestPointCorrectness::test_resize[dtype=int32] SKIPPED [ 23%]
tests/test_augmentations.py::TestPointCorrectness::test_resize[dtype=float32] SKIPPED [ 24%]
tests/test_bindings.py::test_binding[jax] SKIPPED (Skipping non-benc...) [ 25%]
tests/test_bindings.py::test_binding[pytorch] SKIPPED (Skipping non-...) [ 25%]
tests/test_compression.py::test_prepare_dense_features_jpg_only SKIPPED  [ 26%]
tests/test_compression.py::test_compress_many_files[fp16] SKIPPED (S...) [ 27%]
tests/test_compression.py::test_compress_many_files[fp32] SKIPPED (S...) [ 28%]
tests/test_compression.py::test_compress_many_files[fp16_permute] SKIPPED [ 29%]
tests/test_compression.py::test_compress_many_files[fp32_permute] SKIPPED [ 29%]
tests/test_compression.py::test_compress_many_files[fp16_permute_bitshuffle] SKIPPED [ 30%]
tests/test_compression.py::test_compress_many_files[fp16_permute_bitshuffle_compress] SKIPPED [ 31%]
tests/test_compression.py::TestBenchmarks::test_compress[fp32_zstd7] PASSED [ 32%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_zstd7] PASSED [ 32%]
tests/test_compression.py::TestBenchmarks::test_compress[fp32_bshuf_zstd7] PASSED [ 33%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd7] PASSED [ 34%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd(3,7,22)] PASSED [ 35%]
tests/test_compression.py::TestBenchmarks::test_compress[fp16_bshuf_zstd22] PASSED [ 35%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp32_zstd7] PASSED [ 36%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_zstd7] PASSED [ 37%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp32_bshuf_zstd7] PASSED [ 38%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd7] PASSED [ 38%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd(3,7,22)] PASSED [ 39%]
tests/test_compression.py::TestBenchmarks::test_decompress[fp16_bshuf_zstd22] PASSED [ 40%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=16,prefetch=16] SKIPPED [ 41%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=16,prefetch=4] SKIPPED [ 41%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=16] SKIPPED [ 42%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=1] SKIPPED [ 43%]
tests/test_dataloader.py::TestGeneral::test_get_length[threads=8,prefetch=2] SKIPPED [ 44%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=16,prefetch=16] SKIPPED [ 45%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=16,prefetch=4] SKIPPED [ 45%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=16] SKIPPED [ 46%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=1] SKIPPED [ 47%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_once[threads=8,prefetch=2] SKIPPED [ 48%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=16,prefetch=16] SKIPPED [ 48%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=16,prefetch=4] SKIPPED [ 49%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=16] SKIPPED [ 50%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=1] SKIPPED [ 51%]
tests/test_dataloader.py::TestRaster::test_one_dataloader_trice[threads=8,prefetch=2] SKIPPED [ 51%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=16,prefetch=16] SKIPPED [ 52%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=16,prefetch=4] SKIPPED [ 53%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=16] SKIPPED [ 54%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=1] SKIPPED [ 54%]
tests/test_dataloader.py::TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=2] SKIPPED [ 55%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=16,prefetch=16] SKIPPED [ 56%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=16,prefetch=4] SKIPPED [ 57%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=16] SKIPPED [ 58%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=1] SKIPPED [ 58%]
tests/test_dataloader.py::TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=2] SKIPPED [ 59%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=16] SKIPPED [ 60%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=4] SKIPPED [ 61%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=16] SKIPPED [ 61%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=1] SKIPPED [ 62%]
tests/test_dataloader.py::TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=2] SKIPPED [ 63%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=16,prefetch=16] SKIPPED [ 64%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=16,prefetch=4] SKIPPED [ 64%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=16] SKIPPED [ 65%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=1] SKIPPED [ 66%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=2] SKIPPED [ 67%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=16,prefetch=16] SKIPPED [ 67%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=16,prefetch=4] SKIPPED [ 68%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=16] SKIPPED [ 69%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=1] SKIPPED [ 70%]
tests/test_dataloader.py::TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=2] SKIPPED [ 70%]
tests/test_dataloader.py::TestRaster::test_aug_pipe_buffers_upsize SKIPPED [ 71%]
tests/test_dataloader.py::TestDecoders::test_exr SKIPPED (Skipping n...) [ 72%]
tests/test_dataloader.py::TestDecoders::test_png SKIPPED (Skipping n...) [ 73%]
tests/test_dataloader.py::TestDecoders::test_npy[float32] SKIPPED (S...) [ 74%]
tests/test_dataloader.py::TestDecoders::test_compressed SKIPPED (Ski...) [ 74%]
tests/test_dataloader.py::TestPoints::test_points_must_follow_raster_item SKIPPED [ 75%]
tests/test_dataloader.py::TestPoints::test_points_batch_returns_lengths_metadata SKIPPED [ 76%]
tests/test_dataloader.py::TestPoints::test_points_tensor_prefix_matches_lengths SKIPPED [ 77%]
tests/test_dataloader.py::TestPoints::test_points_metadata_matches_dataset_iteration_order SKIPPED [ 77%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=16] PASSED [ 78%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=4] PASSED [ 79%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=2] PASSED [ 80%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=1] PASSED [ 80%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=16] PASSED [ 81%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=4] PASSED [ 82%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=2] PASSED [ 83%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=1] PASSED [ 83%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=16] PASSED [ 84%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=4] PASSED [ 85%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=2] PASSED [ 86%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=1] PASSED [ 87%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=16] PASSED [ 87%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=4] PASSED [ 88%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=2] PASSED [ 89%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=1] PASSED [ 90%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=16] PASSED [ 90%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=4] PASSED [ 91%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=2] PASSED [ 92%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=1] PASSED [ 93%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=16] PASSED [ 93%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=4] PASSED [ 94%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=2] PASSED [ 95%]
tests/test_dataloader.py::test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=1] PASSED [ 96%]
tests/test_dataset.py::test_get_eroneous_dataset SKIPPED (Skipping n...) [ 96%]
tests/test_dataset.py::test_get_dataset SKIPPED (Skipping non-benchm...) [ 97%]
tests/test_dataset.py::test_dataset_works_with_trailing_slash SKIPPED    [ 98%]
tests/test_dataset.py::test_split_dataset SKIPPED (Skipping non-benc...) [ 99%]
tests/test_meta.py::test_version SKIPPED (Skipping non-benchmark (--...) [100%]
Wrote benchmark data in: <_io.BufferedWriter name='.benchmark.json'>



---------------------------------------------------------------------------------- benchmark 'Compression': 6 tests ---------------------------------------------------------------------------------
Name (time in s)                              Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_compress[fp16_bshuf_zstd7]            2.1246 (1.0)      2.1657 (1.0)      2.1452 (1.0)      0.0205 (1.01)     2.1452 (1.0)      0.0308 (1.05)          1;0  0.4662 (1.0)           3           1
test_compress[fp16_zstd7]                  2.2490 (1.06)     2.4155 (1.12)     2.3178 (1.08)     0.0869 (4.27)     2.2889 (1.07)     0.1249 (4.25)          1;0  0.4314 (0.93)          3           1
test_compress[fp32_bshuf_zstd7]            2.9281 (1.38)     2.9780 (1.38)     2.9503 (1.38)     0.0253 (1.24)     2.9449 (1.37)     0.0374 (1.27)          1;0  0.3389 (0.73)          3           1
test_compress[fp32_zstd7]                  3.4651 (1.63)     3.5043 (1.62)     3.4879 (1.63)     0.0204 (1.0)      3.4942 (1.63)     0.0294 (1.0)           1;0  0.2867 (0.62)          3           1
test_compress[fp16_bshuf_zstd22]           7.9230 (3.73)     8.0301 (3.71)     7.9869 (3.72)     0.0565 (2.77)     8.0077 (3.73)     0.0803 (2.73)          1;0  0.1252 (0.27)          3           1
test_compress[fp16_bshuf_zstd(3,7,22)]     8.4547 (3.98)     8.5063 (3.93)     8.4757 (3.95)     0.0272 (1.33)     8.4660 (3.95)     0.0388 (1.32)          1;0  0.1180 (0.25)          3           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------- benchmark 'Dataloader': 24 tests ---------------------------------------------------------------------------------------------------------
Name (time in ms)                                                              Min                   Max                  Mean             StdDev                Median                IQR            Outliers     OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=4]         854.8818 (1.0)        892.3143 (1.0)        869.6026 (1.0)      13.6397 (1.30)       865.6204 (1.0)      25.6542 (2.38)          5;0  1.1500 (1.0)          10           1
test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=2]         856.5035 (1.00)       965.9318 (1.08)       891.7553 (1.03)     29.5711 (2.81)       887.9578 (1.03)     23.5316 (2.18)          2;1  1.1214 (0.98)         10           1
test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=2]         856.9668 (1.00)       903.4354 (1.01)       882.2271 (1.01)     12.9531 (1.23)       883.9058 (1.02)     10.7955 (1.0)           3;2  1.1335 (0.99)         10           1
test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=16]        865.1969 (1.01)       925.9851 (1.04)       885.7360 (1.02)     19.4695 (1.85)       882.3686 (1.02)     27.1877 (2.52)          2;0  1.1290 (0.98)         10           1
test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=16]        866.6836 (1.01)       995.3330 (1.12)       911.0845 (1.05)     36.7786 (3.50)       904.1949 (1.04)     40.2770 (3.73)          2;1  1.0976 (0.95)         10           1
test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=4]         866.9338 (1.01)       949.5620 (1.06)       897.7484 (1.03)     32.5399 (3.10)       889.9900 (1.03)     64.9257 (6.01)          3;0  1.1139 (0.97)         10           1
test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=1]         872.6046 (1.02)       923.8296 (1.04)       892.6462 (1.03)     15.4702 (1.47)       894.7524 (1.03)     21.3980 (1.98)          3;0  1.1203 (0.97)         10           1
test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=1]         873.4055 (1.02)       903.1074 (1.01)       890.1846 (1.02)     10.5111 (1.0)        892.1814 (1.03)     15.9536 (1.48)          3;0  1.1234 (0.98)         10           1
test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=16]       881.4842 (1.03)       998.8921 (1.12)       934.2067 (1.07)     41.2283 (3.92)       934.4204 (1.08)     68.3049 (6.33)          5;0  1.0704 (0.93)         10           1
test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=4]        884.4913 (1.03)     1,036.2951 (1.16)       951.3505 (1.09)     39.6357 (3.77)       945.2845 (1.09)     14.6417 (1.36)          2;3  1.0511 (0.91)         10           1
test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=1]        894.3871 (1.05)     1,105.9667 (1.24)       940.6244 (1.08)     62.4173 (5.94)       925.7966 (1.07)     25.7317 (2.38)          1;2  1.0631 (0.92)         10           1
test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=16]        906.2885 (1.06)     1,033.8145 (1.16)       948.0797 (1.09)     34.7584 (3.31)       938.7160 (1.08)     28.6968 (2.66)          2;1  1.0548 (0.92)         10           1
test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=2]         909.6591 (1.06)     1,004.4194 (1.13)       938.6038 (1.08)     28.1786 (2.68)       930.9284 (1.08)     13.1093 (1.21)          3;2  1.0654 (0.93)         10           1
test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=1]         917.4994 (1.07)       992.1108 (1.11)       949.4047 (1.09)     23.8411 (2.27)       942.0425 (1.09)     23.1275 (2.14)          3;0  1.0533 (0.92)         10           1
test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=4]         919.6601 (1.08)     1,029.8735 (1.15)       956.3095 (1.10)     38.6298 (3.68)       946.9467 (1.09)     14.4416 (1.34)          2;2  1.0457 (0.91)         10           1
test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=2]        928.9227 (1.09)     1,015.6114 (1.14)       959.6468 (1.10)     26.4510 (2.52)       952.5558 (1.10)     34.0316 (3.15)          3;0  1.0421 (0.91)         10           1
test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=1]        957.4651 (1.12)     1,067.6443 (1.20)     1,030.7175 (1.19)     38.4364 (3.66)     1,050.7869 (1.21)     59.7709 (5.54)          2;0  0.9702 (0.84)         10           1
test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=2]        972.9366 (1.14)     1,089.1745 (1.22)     1,013.0144 (1.16)     38.7777 (3.69)     1,006.9103 (1.16)     44.1704 (4.09)          3;1  0.9872 (0.86)         10           1
test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=4]        974.5629 (1.14)     1,117.9892 (1.25)     1,044.4552 (1.20)     45.8739 (4.36)     1,044.1351 (1.21)     76.5114 (7.09)          4;0  0.9574 (0.83)         10           1
test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=16]       980.1058 (1.15)     1,176.6177 (1.32)     1,054.9527 (1.21)     62.7622 (5.97)     1,033.3070 (1.19)     90.8968 (8.42)          3;0  0.9479 (0.82)         10           1
test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=16]       992.2311 (1.16)     1,092.0902 (1.22)     1,043.0542 (1.20)     34.9778 (3.33)     1,048.7797 (1.21)     43.1283 (4.00)          4;0  0.9587 (0.83)         10           1
test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=1]      1,007.1982 (1.18)     1,114.4916 (1.25)     1,066.9164 (1.23)     40.0228 (3.81)     1,076.1850 (1.24)     65.5778 (6.07)          3;0  0.9373 (0.82)         10           1
test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=2]      1,011.5120 (1.18)     1,182.0017 (1.32)     1,088.7925 (1.25)     53.8138 (5.12)     1,096.9589 (1.27)     96.0361 (8.90)          4;0  0.9184 (0.80)         10           1
test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=4]      1,030.2588 (1.21)     1,185.2402 (1.33)     1,091.0991 (1.25)     42.3633 (4.03)     1,093.0836 (1.26)     34.7820 (3.22)          3;1  0.9165 (0.80)         10           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------- benchmark 'Decompression': 6 tests ---------------------------------------------------------------------------------
Name (time in s)                                Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_decompress[fp16_bshuf_zstd7]            2.5991 (1.0)      2.6071 (1.0)      2.6043 (1.0)      0.0045 (1.68)     2.6068 (1.0)      0.0060 (1.62)          1;0  0.3840 (1.0)           3           1
test_decompress[fp16_bshuf_zstd(3,7,22)]     2.6293 (1.01)     2.6391 (1.01)     2.6332 (1.01)     0.0052 (1.92)     2.6313 (1.01)     0.0073 (1.99)          1;0  0.3798 (0.99)          3           1
test_decompress[fp16_zstd7]                  2.6310 (1.01)     2.6359 (1.01)     2.6328 (1.01)     0.0027 (1.02)     2.6314 (1.01)     0.0037 (1.0)           1;0  0.3798 (0.99)          3           1
test_decompress[fp16_bshuf_zstd22]           2.6414 (1.02)     2.6464 (1.02)     2.6444 (1.02)     0.0027 (1.0)      2.6454 (1.01)     0.0038 (1.03)          1;0  0.3782 (0.98)          3           1
test_decompress[fp32_zstd7]                  2.8631 (1.10)     2.8823 (1.11)     2.8718 (1.10)     0.0097 (3.62)     2.8700 (1.10)     0.0144 (3.90)          1;0  0.3482 (0.91)          3           1
test_decompress[fp32_bshuf_zstd7]            2.8759 (1.11)     2.8822 (1.11)     2.8796 (1.11)     0.0033 (1.22)     2.8807 (1.11)     0.0047 (1.27)          1;0  0.3473 (0.90)          3           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
================== 36 passed, 95 skipped in 567.16s (0:09:27) ==================
```
