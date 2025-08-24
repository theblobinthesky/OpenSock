from enum import Enum
from typing import Callable, List, Tuple
from . import _core as m
import jax, jax.numpy as jnp
import numpy as np


class FileType(Enum):
    JPG = m.FileType.JPG
    PNG = m.FileType.PNG
    EXR = m.FileType.EXR
    NPY = m.FileType.NPY
    COMPRESSED = m.FileType.COMPRESSED


class ItemFormat(Enum):
    FLOAT = m.ItemFormat.FLOAT
    UINT = m.ItemFormat.UINT


class Head:
    def __init__(self, file_type: FileType, dict_name: str,
                 shape: Tuple[int, ...]):
        """Initialize a head descriptor for the dataset."""
        self._native = m.Head(file_type.value, dict_name, list(shape))

    @staticmethod
    def from_native(native_head) -> 'Head':
        """Wrap a native head instance."""
        obj = Head.__new__(Head)
        obj._native = native_head
        return obj

    def get_ext(self) -> str:
        """Return the file extension associated with this head."""
        return self._native.getExt()

    def dict_name(self) -> str:
        """Return the name identifier of this head."""
        return self._native.getDictName()

    def shape(self) -> List[int]:
        """Return the dimensions defined for this head."""
        return self._native.getShape()

    def shape_size(self) -> int:
        """Return the total number of elements given the shape dimensions."""
        return self._native.getShapeSize()

    def file_type(self) -> FileType:
        """Return the file type as a FileType enum."""
        native_ft = self._native.getFilesType()
        mapping = {
            m.FileType.JPG: FileType.JPG,
            m.FileType.PNG: FileType.PNG,
            m.FileType.EXR: FileType.EXR,
            m.FileType.NPY: FileType.NPY,
            m.FileType.COMPRESSED: FileType.COMPRESSED,
        }
        return mapping[native_ft]

    def item_format(self) -> ItemFormat:
        """Return the item format as a ItemFormat enum."""
        native_if = self._native.getItemFormat()
        if native_if == m.ItemFormat.FLOAT:
            return ItemFormat.FLOAT
        elif native_if == m.ItemFormat.UINT:
            return ItemFormat.UINT
        else:
            raise ValueError("Unknown item format")

    def bytes_per_item(self) -> int:
        """Return the number of bytes per item as an integer."""
        return self._native.getBytesPerItem()


class Dataset:
    def __init__(self, native_dataset, post_process_fn):
        self._native = native_dataset
        self.post_process_fn = post_process_fn

    @classmethod
    def from_subdirs(cls, root_dir: str, heads: List[Head],
                     sub_dirs: List[str],
                     create_dataset_function: Callable,
                     post_process_fn: Callable[[dict[str, jnp.ndarray]], dict[str, jnp.ndarray]] = None,
                     is_virtual_dataset: bool = False) -> 'Dataset':
        """
        Construct a Dataset from subdirectories.
        The heads should be supplied in the order they will be used in.
        """
        if post_process_fn is None:
            post_process_fn = lambda x: x

        native = m.Dataset(root_dir, [h._native for h in heads],
                           sub_dirs, create_dataset_function, is_virtual_dataset)
        return cls(native, post_process_fn)

    @classmethod
    def from_entries(cls, root_dir: str, heads: List[Head],
                     entries: List[List[str]],
                     post_process_fn: Callable[[dict[str, jnp.ndarray]], dict[str, jnp.ndarray]] = None) -> 'Dataset':
        """
        Construct a Dataset from entries. Entries may or may not be preceeded by the root directory.
        The heads should be supplied in the order they will be used in.
        """
        native = m.Dataset(root_dir, [h._native for h in heads], entries)
        return cls(native, post_process_fn)

    def split_train_validation_test(self, train_percentage: float,
                                    valid_percentage: float
                                    ) -> Tuple['Dataset', 'Dataset', 'Dataset']:
        """Split the dataset into training, validation, and test subsets."""
        train, valid, test = self._native.splitTrainValidationTest(train_percentage, valid_percentage)
        return Dataset(train, self.post_process_fn), Dataset(valid, self.post_process_fn), Dataset(test,
                                                                                                   self.post_process_fn)

    @property
    def root_dir(self) -> str:
        """Return the root directory of the dataset."""
        return self._native.getRootDir()

    @property
    def heads(self) -> List[Head]:
        """Return a list of head descriptors for the dataset."""
        return [Head.from_native(h) for h in self._native.getHeads()]

    @property
    def entries(self) -> List[List[str]]:
        """Return the dataset entries."""
        return self._native.getEntries()

    def __len__(self) -> int:
        return len(self._native.getEntries())


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
        self.head_dict = {h.dict_name(): h for h in self.dataset.heads}

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._native)

    def get_next_batch(self) -> dict:
        """Return next batch as a dict of jax arrays."""

        def from_dlpack(x):
            return jax.dlpack.from_dlpack(x, copy=False)

        batch = self._native.getNextBatch()
        batch = {key: from_dlpack(x) for key, x in batch.items()}

        # Cast JPG entries to float32 and normalize
        for key in list(batch.keys()):
            ft = self.head_dict[key].file_type()
            if ft == FileType.JPG or ft == FileType.PNG:
                batch[key] = batch[key].astype(jnp.float32) / 255.0

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
    def __init__(self, shape: np.ndarray):
        """Wrapper around the native Decompressor."""
        self._native = m.Decompressor(shape)

    def decompress(self, path: str) -> np.ndarray:
        return self._native.decompress(path)


def shutdown_resource_pool() -> None:
    """Shut down the global resource pool and free GPU/host memory.

    Stops internal threads, waits for in-flight buffers to be returned,
    and releases associated CUDA resources. Safe to call multiple times.
    The resource pool is also shut down automatically at interpreter exit.
    """
    m.shutdown_resource_pool()
