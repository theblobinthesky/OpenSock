from enum import Enum
from typing import Callable, Dict, List, Optional
from . import _core as m
import jax
import jax.numpy as jnp


class ItemFormat(Enum):
    FLOAT = m.ItemFormat.FLOAT
    UINT = m.ItemFormat.UINT


class Dataset:
    def __init__(self, native_dataset, post_process_fn: Optional[Callable] = None):
        self._native = native_dataset
        self.post_process_fn = post_process_fn

    @classmethod
    def from_subdirs(cls,
                     root_dir: str,
                     subdir_to_dict: Dict[str, str],
                     create_dataset_function: Callable,
                     post_process_fn: Optional[Callable[[dict], dict]] = None,
                     is_virtual_dataset: bool = False) -> 'Dataset':
        native = m.Dataset(
            m.FlatDataSource(root_dir, subdir_to_dict),
            create_dataset_function,
            is_virtual_dataset
        )
        return cls(native, post_process_fn)

    def split_train_validation_test(self, train_percentage: float,
                                    valid_percentage: float):
        train, valid, test = self._native.splitTrainValidationTest(train_percentage, valid_percentage)
        return Dataset(train, self.post_process_fn), Dataset(valid, self.post_process_fn), Dataset(
            test, self.post_process_fn)

    @property
    def entries(self) -> List[List[str]]:
        """Return the dataset entries."""
        return self._native.getEntries()

    def __len__(self) -> int:
        return len(self.entries)


class BatchedDataset:
    def __init__(self, dataset: Dataset, batch_size: int):
        self._native = m.BatchedDataset(dataset._native, batch_size)

    def get_next_batch(self) -> List[List[str]]:
        """Return the next batch of file entries."""
        return self._native.getNextBatch()


class JaxSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        # Ensure Jax is using GPU; otherwise raise an error.
        device = jax.devices('gpu')[0]

        def jax_has_gpu():
            try:
                _ = jax.device_put(jax.numpy.ones(1), device=device)
                return True
            except Exception:
                return False

        if jax.default_backend() != 'gpu' or not jax_has_gpu():
            raise EnvironmentError('Jax is not using the GPU!')

        x = jnp.zeros((16, 16))
        x = jax.device_put(x, device)
        assert x.mean() == 0.0


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int,
                 num_threads: int = 4,
                 prefetch_size: int = 4):
        """Wrap native DataLoader and ensure Jax GPU is initialized."""
        JaxSingleton()
        self.dataset = dataset
        self._native = m.DataLoader(dataset._native, batch_size, num_threads, prefetch_size)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._native)

    def get_next_batch(self) -> dict:
        """Return next batch as a dict of jax arrays."""

        def from_dlpack(x):
            return jax.dlpack.from_dlpack(x, copy=False)

        batch = self._native.getNextBatch()
        batch = {key: from_dlpack(x) for key, x in batch.items()}

        # Normalize uint8 inputs to float32 in [0,1]; leave other dtypes intact.
        for key, arr in list(batch.items()):
            if arr.dtype == jnp.uint8:
                batch[key] = arr.astype(jnp.float32) / 255.0

        return self.dataset.post_process_fn(batch)


class Codec(Enum):
    ZSTD_LEVEL_3 = m.Codec.ZSTD_LEVEL_3
    ZSTD_LEVEL_7 = m.Codec.ZSTD_LEVEL_7
    ZSTD_LEVEL_22 = m.Codec.ZSTD_LEVEL_22


class CompressorOptions:
    def __init__(self,
                 num_threads: int,
                 input_directory: str,
                 output_directory: str,
                 shape: List[int],
                 cast_to_fp16: bool,
                 permutations: List[List[int]],
                 with_bitshuffle: bool,
                 allowed_codecs: List[Codec],
                 tolerance_for_worse_codec: float):
        """
        Options for configuring the Compressor.
        """
        self._native = m.CompressorOptions(
            num_threads,
            input_directory,
            output_directory,
            shape,
            cast_to_fp16,
            permutations,
            with_bitshuffle,
            [c.value for c in allowed_codecs],
            tolerance_for_worse_codec
        )


class Compressor:
    def __init__(self, options: CompressorOptions):
        """Wrapper around the native Compressor."""
        self._native = m.Compressor(options._native)

    def start(self):
        """Begin the compression process."""
        return self._native.start()


class Decompressor:
    def __init__(self, shape: List[int]):
        """Wrapper around the native Decompressor."""
        self._native = m.Decompressor(shape)

    def decompress(self, path: str):
        return self._native.decompress(path)


def shutdown_resource_pool() -> None:
    """Shut down the global resource pool and free GPU/host memory."""
    m.shutdown_resource_pool()
