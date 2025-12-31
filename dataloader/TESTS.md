# Test Report

Generated: 2025-12-31 21:10:48

## Normal

Command: `pytest ./tests/test_dataloader.py ./tests/test_bindings.py --benchmark-disable`

- passed: 50
- failed: 2
- errors: 0
- skipped: 0

### Results

- ✗ tests.test_dataloader.TestRaster::test_one_dataloader_once[threads=8,prefetch=16]: AssertionError: Error 74.0 too high for image
assert np.False_
 +  where np.False_ = <function all at 0x734338395f70>(np.float32(0.29002872) < (10 / 255.0))
 +    where <function all at 0x734338395f70> = np.all
- ✗ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=16]: AssertionError: Error 72.8 too high for image
assert np.False_
 +  where np.False_ = <function all at 0x734338395f70>(np.float32(0.28552464) < (10 / 255.0))
 +    where <function all at 0x734338395f70> = np.all
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
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=16,prefetch=4]
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=16]
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=1]
- ✓ tests.test_dataloader.TestRaster::test_two_dlers_with_different_batch_sizes[threads=8,prefetch=2]
- ✓ tests.test_dataloader::test_end_to_end_perf

## Sanitizers (ASan/USan)

Command: `pytest tests/test_meta.py tests/test_dataset.py tests/test_augmentations.py tests/test_compression.py -xs --benchmark-disable (with ASan/USan)`

- passed: 0
- failed: 0
- errors: 1
- skipped: 0

### Results


