import pytest
import numpy as np
import cv2
from native_dataloader import (
    FlipAugmentation, 
    PadAugmentation, 
    RandomCropAugmentation, 
    ResizeAugmentation, 
    DataAugmentationPipe, 
    PadSettings
)

NUM_EXAMPLES = 64
DEF_MAX_INPUT_SHAPE = [4, 128, 128, 3]
DEF_MAX_NUM_POINTS = 128
DEF_MAX_ITEM_SIZE = 4

def def_get_processing_schema(pipe, shape, seed):
    shapes = [list(shape[1:]) for _ in range(shape[0])]
    return pipe.get_processing_schema(shapes, seed) # TODO: Mixed by batch!

def get_():
    augmentations = [ResizeAugmentation(32, 32)]
    pipe = DataAugmentationPipe(augmentations, DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, DEF_MAX_ITEM_SIZE)

    input_shape = [4, 32, 32, 3]
    seed = 123
    output_shape, schema_capsule = def_get_processing_schema(pipe, input_shape, seed)

    return pipe, output_shape, schema_capsule

def gen_random(shape, dtype):
    return (np.random.rand(*shape) * 1024).astype(dtype)

@pytest.fixture(params=['uint8', 'int32', 'float32'], ids=lambda dtype: f"dtype={dtype}")
def supported_dtypes_config(request):
    return {"dtype": request.param}

@pytest.fixture(params=['uint8', 'float32'], ids=lambda dtype: f"dtype={dtype}")
def supported_dtypes_config_no_ints(request):
    return {"dtype": request.param}

@pytest.fixture(params=['int32', 'float32'], ids=lambda dtype: f"dtype={dtype}")
def supported_signed_dtypes_config(request):
    return {"dtype": request.param}


class TestBasics:
    def test_augment_raster_executes_successfully(self):
        pipe, output_shape, schema_capsule = get_()
        input_array = gen_random((4, 32, 32, 3), np.float32)
        output_array = np.zeros(output_shape, dtype=np.float32)
        pipe.augment_raster(input_array, output_array, schema_capsule)

    def test_augment_raster_fails_with_incorrect_shape(self):
        pipe, output_shape, schema_capsule = get_()
        input_array = gen_random((5, 32, 32, 3), np.float32)
        output_array = np.zeros(output_shape, dtype=np.float32)

        with pytest.raises(RuntimeError):
            pipe.augment_raster(input_array, output_array, schema_capsule)

    def test_augment_raster_fails_with_mismatched_dtype(self):
        pipe, output_shape, schema_capsule = get_()
        input_array = gen_random((4, 32, 32, 3), np.float32)
        output_array = np.zeros(output_shape, dtype=np.float64)

        with pytest.raises(RuntimeError):
            pipe.augment_raster(input_array, output_array, schema_capsule)

    def test_augment_raster_succeedes_for_supported_dtype(self, supported_dtypes_config):
        dtype = supported_dtypes_config["dtype"]
        pipe, output_shape, schema_capsule = get_()
        input_array = gen_random((4, 32, 32, 3), dtype)
        output_array = np.zeros(output_shape, dtype=dtype)
        pipe.augment_raster(input_array, output_array, schema_capsule)

    def test_augment_raster_fails_for_unsupported_dtype(self):
        pipe, output_shape, schema_capsule = get_()
        input_array = gen_random((4, 32, 32, 3), np.uint64)
        output_array = np.zeros(output_shape, dtype=np.uint64)

        with pytest.raises(RuntimeError):
            pipe.augment_raster(input_array, output_array, schema_capsule)

    def test_pipe_fails_with_no_augmentation(self):
        augmentations = []
        with pytest.raises(RuntimeError):
            DataAugmentationPipe(augmentations, DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, DEF_MAX_ITEM_SIZE)

    BR = PadSettings.PAD_BOTTOM_RIGHT
    @pytest.mark.parametrize("augs", [
            [PadAugmentation(128, 128, BR)],
            [PadAugmentation(128, 128, BR), PadAugmentation(128, 256, BR)],
            [PadAugmentation(128, 256, BR), PadAugmentation(128, 256, BR)],
            [PadAugmentation(128, 128, BR), PadAugmentation(128, 128, BR)],
            [PadAugmentation(128, 256, BR), PadAugmentation(128, 256, BR), PadAugmentation(256, 256, BR)]
        ], ids=["1_skip", "1_skip_2_swap", "1_swap_2_skip", "1_skip_2_skip", "1_swap_2_skip_3_swap"])
    def test_augmentations_are_skipped(self, augs):
        input_shape = (4, 128, 128, 3)
        pipe = DataAugmentationPipe(augs, DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, 8)
        output_shape, schema_capsule = def_get_processing_schema(pipe, input_shape, 0)
        input_array = gen_random(input_shape, np.float32)
        output_array = np.zeros(output_shape, dtype=np.float32)
        pipe.augment_raster(input_array, output_array, schema_capsule)
        assert np.all(input_array == output_array[:, :128, :128])


class TestAugmentationRules:
    def test_random_crop_fails_with_different_min_max(self):
        augmentations = [RandomCropAugmentation(5, 10, 10, 10)]
        with pytest.raises(RuntimeError):
            DataAugmentationPipe(augmentations, DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, DEF_MAX_ITEM_SIZE)

    def test_random_crop_succeedes_with_same_min_max(self):
        augmentations = [RandomCropAugmentation(10, 10, 10, 10)]
        DataAugmentationPipe(augmentations, DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, DEF_MAX_ITEM_SIZE)

    def test_flip_fails_if_disabled(self):
        with pytest.raises(RuntimeError):
            FlipAugmentation(0.0, 0.0)

    def test_flip_tries_different_settings(self):
        configs = set()
        aug = FlipAugmentation(0.5, 0.5)
        input_shape = (4, 128, 64, 3)
        for item_seed in range(NUM_EXAMPLES):
            settings = aug.get_item_settings(input_shape, item_seed)
            configs.add((settings['does_vertical_flip'], settings['does_horizontal_flip']))
        assert len(configs) > 2

    # TODO: Current limitation. Will be removed once resize switches to our own interpolation.
    def test_resize_fails_if_not_three_channels(self):
        with pytest.raises(RuntimeError):
            DataAugmentationPipe([ResizeAugmentation(48, 48)], (4, 128, 128, 2), DEF_MAX_NUM_POINTS, 8)


class TestRasterCorrectness:
    def test_flip(self, supported_dtypes_config):
        dtype = supported_dtypes_config["dtype"]
        aug = FlipAugmentation(0.5, 0.5)
        input_shape = (4, 128, 64, 3)
        pipe = DataAugmentationPipe(
            [aug, PadAugmentation(128, 128, PadSettings.PAD_BOTTOM_RIGHT)], 
            DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, 8
        )

        for item_seed in range(NUM_EXAMPLES):
            settings = aug.get_item_settings(input_shape, item_seed)
            output_shape, schema_capsule = def_get_processing_schema(pipe, input_shape, item_seed)
            input_array = gen_random(input_shape, dtype)
            output_array = np.zeros(output_shape, dtype=dtype)
            pipe.augment_raster(input_array, output_array, schema_capsule)

            gt = input_array
            if settings['does_horizontal_flip']: gt = gt[:, :, ::-1]
            if settings['does_vertical_flip']: gt = gt[:, ::-1]
            assert np.all(gt == output_array[:, :, :64])

    def test_random_crop(self, supported_dtypes_config):
        dtype = supported_dtypes_config["dtype"]
        aug = RandomCropAugmentation(16, 24, 32, 48)
        input_shape = (4, 128, 64, 3)
        pipe = DataAugmentationPipe(
            [aug, PadAugmentation(128, 128, PadSettings.PAD_BOTTOM_RIGHT)], 
           DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, 8
        )

        for item_seed in range(NUM_EXAMPLES):
            settings = aug.get_item_settings(input_shape[1:], item_seed)
            output_shape, schema_capsule = def_get_processing_schema(pipe, input_shape, item_seed)
            input_array = gen_random(input_shape, dtype)
            output_array = np.zeros(output_shape, dtype=dtype)
            pipe.augment_raster(input_array, output_array, schema_capsule)

            t, l, h, w = settings['top'], settings['left'], settings['height'], settings['width']
            assert np.all(input_array[:, t:t+h, l:l+w] == output_array[:, :h, :w])

    def test_resize(self, supported_dtypes_config_no_ints):
        B = 4
        dtype = supported_dtypes_config_no_ints["dtype"]
        input_shape = (B, 128, 64, 3)
        pipe = DataAugmentationPipe([ResizeAugmentation(48, 36)], DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, 8)

        for item_seed in range(NUM_EXAMPLES):
            output_shape, schema_capsule = def_get_processing_schema(pipe, input_shape, item_seed)
            i, j = np.meshgrid(np.arange(128), np.arange(64), indexing='ij')
            part = np.stack([i, j, i + j], axis=-1)[np.newaxis, ...]
            input_array = np.repeat(part, B, axis=0).astype(dtype)
            output_array = np.zeros(output_shape, dtype=dtype)

            pipe.augment_raster(input_array, output_array, schema_capsule)
            gt = np.stack([cv2.resize(input_array[i], (36, 48)) for i in range(B)], axis=0)

            gt = gt.astype(np.double)
            output_array = output_array.astype(np.double)
            assert np.mean(np.abs(gt - output_array)) < 0.5


class TestPointCorrectness:
    def test_flip(self, supported_signed_dtypes_config):
        dtype = supported_signed_dtypes_config["dtype"]
        aug = FlipAugmentation(0.5, 0.5)
        h, w = 128, 64
        input_shape, points_shape = (4, h, w, 3), (4, 128, 2)
        pipe = DataAugmentationPipe(
            [aug, PadAugmentation(128, 128, PadSettings.PAD_BOTTOM_RIGHT)], 
            DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, 8
        )

        for item_seed in range(NUM_EXAMPLES):
            settings = aug.get_item_settings(input_shape, item_seed)
            _, schema_capsule = def_get_processing_schema(pipe, input_shape, item_seed)
            input_array = gen_random(points_shape, dtype)
            output_array = np.zeros(points_shape, dtype=dtype)
            pipe.augment_points(input_array, output_array, schema_capsule)

            gt = input_array
            if settings['does_vertical_flip']: gt[:, :, 0] = h - 1 - gt[:, :, 0]
            if settings['does_horizontal_flip']: gt[:, :, 1] = w - 1 - gt[:, :, 1]
            assert np.all(gt == output_array)

    def test_random_crop(self, supported_signed_dtypes_config):
        dtype = supported_signed_dtypes_config["dtype"]
        aug = RandomCropAugmentation(16, 24, 32, 48)
        input_shape, points_shape = (4, 128, 64, 3), (4, 128, 2)
        pipe = DataAugmentationPipe(
            [aug, PadAugmentation(128, 128, PadSettings.PAD_BOTTOM_RIGHT)], 
            DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, 8
        )

        for item_seed in range(NUM_EXAMPLES):
            settings = aug.get_item_settings(input_shape[1:], item_seed)
            _, schema_capsule = def_get_processing_schema(pipe, input_shape, item_seed)
            input_array = gen_random(points_shape, dtype)
            output_array = np.zeros(points_shape, dtype=dtype)
            pipe.augment_points(input_array, output_array, schema_capsule)

            t, l = settings['top'], settings['left']
            gt = input_array
            gt[:, :, 0] -= t
            gt[:, :, 1] -= l
            assert np.all(gt == output_array)

    def test_resize(self, supported_signed_dtypes_config):
        B = 4
        dtype = supported_signed_dtypes_config["dtype"]
        input_shape, points_shape = (B, 128, 64, 3), (4, 128, 2)
        pipe = DataAugmentationPipe([ResizeAugmentation(48, 36)], DEF_MAX_INPUT_SHAPE, DEF_MAX_NUM_POINTS, 8)

        for item_seed in range(NUM_EXAMPLES):
            _, schema_capsule = def_get_processing_schema(pipe, input_shape, item_seed)
            input_array = gen_random(points_shape, dtype)
            output_array = np.zeros(points_shape, dtype=dtype)
            pipe.augment_points(input_array, output_array, schema_capsule)

            gt = input_array.astype(np.float64)
            gt[:, :, 0] *= 48.0 / 128
            gt[:, :, 1] *= 36.0 / 64
            gt = gt.astype(dtype)
            assert np.allclose(gt, output_array)
