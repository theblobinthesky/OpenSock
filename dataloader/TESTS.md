# Test Report

Generated: 2026-01-02 17:39:56

## Normal

Command: `pytest ./tests/test_dataloader.py ./tests/test_bindings.py --benchmark-skip`

- passed: 51
- failed: 0
- errors: 0
- skipped: 24

### Results

- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=16]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=1]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=2]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=16,num_threads=16,prefetch_size=4]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=16]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=1]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=2]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=16,num_threads=8,prefetch_size=4]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=16]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=1]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=2]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=32,num_threads=16,prefetch_size=4]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=16]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=1]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=2]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=32,num_threads=8,prefetch_size=4]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=16]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=1]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=2]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=64,num_threads=16,prefetch_size=4]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=16]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=1]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=2]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_dataloader::test_end_to_end_perf[batch_size=64,num_threads=8,prefetch_size=4]: Skipping benchmark (--benchmark-skip active).
- ✓ tests.test_bindings::test_binding[jax]
- ✓ tests.test_bindings::test_binding[pytorch]
- ✓ tests.test_dataloader.TestDecoders::test_compressed
- ✓ tests.test_dataloader.TestDecoders::test_exr
- ✓ tests.test_dataloader.TestDecoders::test_npy[float32]
- ✓ tests.test_dataloader.TestDecoders::test_png
- ✓ tests.test_dataloader.TestGeneral::test_get_length[threads=16,prefetch=16]
- ✓ tests.test_dataloader.TestGeneral::test_get_length[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestGeneral::test_get_length[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestGeneral::test_get_length[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestGeneral::test_get_length[threads=8,prefetch=2]
- ✓ tests.test_dataloader.TestPoints::test_points_batch_returns_lengths_metadata
- ✓ tests.test_dataloader.TestPoints::test_points_metadata_matches_dataset_iteration_order
- ✓ tests.test_dataloader.TestPoints::test_points_must_follow_raster_item
- ✓ tests.test_dataloader.TestPoints::test_points_tensor_prefix_matches_lengths
- ✓ tests.test_dataloader.TestRaster::test_aug_pipe_buffers_upsize
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=16,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_across_jpg_dataloaders[threads=8,prefetch=2]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=16,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestRaster::test_no_duplicates_within_jpg_dataloaders[threads=8,prefetch=2]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_once[threads=16,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_once[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_once[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_once[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_once[threads=8,prefetch=2]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_trice[threads=16,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_trice[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_trice[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_trice[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestRaster::test_one_dataloader_trice[threads=8,prefetch=2]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_with_next_batch[threads=16,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_with_next_batch[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_with_next_batch[threads=8,prefetch=2]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_without_next_batch[threads=16,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_without_next_batch[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestRaster::test_three_dlers_without_next_batch[threads=8,prefetch=2]
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=2]

## Sanitizers (ASan/USan)

Command: `pytest tests/test_meta.py tests/test_dataset.py tests/test_augmentations.py tests/test_compression.py -xs --benchmark-skip (with ASan/USan)`

- passed: 49
- failed: 0
- errors: 0
- skipped: 12

### Results

- ⚠ tests.test_compression.TestBenchmarks::test_compress[fp16_bshuf_zstd(3,7,22)]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_compress[fp16_bshuf_zstd22]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_compress[fp16_bshuf_zstd7]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_compress[fp16_zstd7]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_compress[fp32_bshuf_zstd7]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_compress[fp32_zstd7]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_decompress[fp16_bshuf_zstd(3,7,22)]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_decompress[fp16_bshuf_zstd22]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_decompress[fp16_bshuf_zstd7]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_decompress[fp16_zstd7]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_decompress[fp32_bshuf_zstd7]: Skipping benchmark (--benchmark-skip active).
- ⚠ tests.test_compression.TestBenchmarks::test_decompress[fp32_zstd7]: Skipping benchmark (--benchmark-skip active).
- ✓ tests.test_augmentations.TestAugmentationRules::test_flip_fails_if_disabled
- ✓ tests.test_augmentations.TestAugmentationRules::test_flip_tries_different_settings
- ✓ tests.test_augmentations.TestAugmentationRules::test_random_crop_fails_with_different_min_max
- ✓ tests.test_augmentations.TestAugmentationRules::test_random_crop_succeedes_with_same_min_max
- ✓ tests.test_augmentations.TestAugmentationRules::test_resize_fails_if_not_three_channels
- ✓ tests.test_augmentations.TestBasics::test_augment_raster_executes_successfully
- ✓ tests.test_augmentations.TestBasics::test_augment_raster_fails_for_unsupported_dtype
- ✓ tests.test_augmentations.TestBasics::test_augment_raster_fails_with_incorrect_shape
- ✓ tests.test_augmentations.TestBasics::test_augment_raster_fails_with_mismatched_dtype
- ✓ tests.test_augmentations.TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=float32]
- ✓ tests.test_augmentations.TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=int32]
- ✓ tests.test_augmentations.TestBasics::test_augment_raster_succeedes_for_supported_dtype[dtype=uint8]
- ✓ tests.test_augmentations.TestBasics::test_augmentations_are_skipped[1_skip]
- ✓ tests.test_augmentations.TestBasics::test_augmentations_are_skipped[1_skip_2_skip]
- ✓ tests.test_augmentations.TestBasics::test_augmentations_are_skipped[1_skip_2_swap]
- ✓ tests.test_augmentations.TestBasics::test_augmentations_are_skipped[1_swap_2_skip]
- ✓ tests.test_augmentations.TestBasics::test_augmentations_are_skipped[1_swap_2_skip_3_swap]
- ✓ tests.test_augmentations.TestBasics::test_pipe_fails_with_dynamic_output_shape[random-crop]
- ✓ tests.test_augmentations.TestBasics::test_pipe_fails_with_dynamic_output_shape[resize/random-crop]
- ✓ tests.test_augmentations.TestBasics::test_pipe_fails_with_no_augmentation
- ✓ tests.test_augmentations.TestBasics::test_pipe_succeedes_with_static_output_shape[random-crop/resize]
- ✓ tests.test_augmentations.TestBasics::test_pipe_succeedes_with_static_output_shape[resize/flip]
- ✓ tests.test_augmentations.TestBasics::test_pipe_succeedes_with_static_output_shape[resize]
- ✓ tests.test_augmentations.TestPointCorrectness::test_flip[dtype=float32]
- ✓ tests.test_augmentations.TestPointCorrectness::test_flip[dtype=int32]
- ✓ tests.test_augmentations.TestPointCorrectness::test_random_crop[dtype=float32]
- ✓ tests.test_augmentations.TestPointCorrectness::test_random_crop[dtype=int32]
- ✓ tests.test_augmentations.TestPointCorrectness::test_resize[dtype=float32]
- ✓ tests.test_augmentations.TestPointCorrectness::test_resize[dtype=int32]
- ✓ tests.test_augmentations.TestRasterCorrectness::test_flip[dtype=float32]
- ✓ tests.test_augmentations.TestRasterCorrectness::test_flip[dtype=int32]
- ✓ tests.test_augmentations.TestRasterCorrectness::test_flip[dtype=uint8]
- ✓ tests.test_augmentations.TestRasterCorrectness::test_random_crop[dtype=float32]
- ✓ tests.test_augmentations.TestRasterCorrectness::test_random_crop[dtype=int32]
- ✓ tests.test_augmentations.TestRasterCorrectness::test_random_crop[dtype=uint8]
- ✓ tests.test_augmentations.TestRasterCorrectness::test_resize[dtype=float32]
- ✓ tests.test_augmentations.TestRasterCorrectness::test_resize[dtype=uint8]
- ✓ tests.test_compression::test_compress_many_files[fp16]
- ✓ tests.test_compression::test_compress_many_files[fp16_permute]
- ✓ tests.test_compression::test_compress_many_files[fp16_permute_bitshuffle]
- ✓ tests.test_compression::test_compress_many_files[fp16_permute_bitshuffle_compress]
- ✓ tests.test_compression::test_compress_many_files[fp32]
- ✓ tests.test_compression::test_compress_many_files[fp32_permute]
- ✓ tests.test_compression::test_prepare_dense_features_jpg_only
- ✓ tests.test_dataset::test_dataset_works_with_trailing_slash
- ✓ tests.test_dataset::test_get_dataset
- ✓ tests.test_dataset::test_get_eroneous_dataset
- ✓ tests.test_dataset::test_split_dataset
- ✓ tests.test_meta::test_version

