from enum import Enum
from typing import Callable, List, Optional, Tuple
from . import _core as m
import jax, jax.numpy as jnp

class FileType(Enum):
    JPG = m.FileType.JPG
    EXR = m.FileType.EXR
    NPY = m.FileType.NPY


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
        if native_ft == m.FileType.JPG:
            return FileType.JPG
        elif native_ft == m.FileType.EXR:
            return FileType.EXR
        elif native_ft == m.FileType.NPY:
            return FileType.NPY
        else:
            raise ValueError("Unknown file type")


class Dataset:
    def __init__(self, native_dataset):
        self._native = native_dataset

    @classmethod
    def from_subdirs(cls, root_dir: str, heads: List[Head],
                     sub_dirs: List[str],
                     create_dataset_function: Callable) -> 'Dataset':
        native = m.Dataset(root_dir, [h._native for h in heads],
                           sub_dirs, create_dataset_function)
        return cls(native)

    @classmethod
    def from_entries(cls, root_dir: str, heads: List[Head],
                     entries: List[List[str]]) -> 'Dataset':
        """Construct a Dataset from entries. Entries may or may not be preceeded by the root directory."""
        native = m.Dataset(root_dir, [h._native for h in heads], entries)
        return cls(native)

    def split_train_validation_test(self, train_percentage: float,
                                    valid_percentage: float
                                    ) -> Tuple['Dataset', 'Dataset', 'Dataset']:
        """Split the dataset into training, validation, and test subsets."""
        train, valid, test = self._native.splitTrainValidationTest(
            train_percentage, valid_percentage)
        return Dataset(train), Dataset(valid), Dataset(test)

    def get_next_batch(self, batch_size: int) -> List[List[str]]:
        """Return the next batch of file entries."""
        return self._native.getNextBatch(batch_size)

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


class DLManagedTensorWrapper:
    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, stream=None):
        return self._capsule

    def __dlpack_device__(self):
        # Return (device_type, device_id); here CUDA=2 and GPU id=0.
        kDLCUDA = 2
        return (kDLCUDA, 0)


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
        self._native = m.DataLoader(dataset._native, batch_size, num_threads, prefetch_size)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._native)

    def get_next_batch(self) -> dict:
        """Return next batch as a dict of jax arrays."""

        def from_dlpack(x):
            return jax.dlpack.from_dlpack(DLManagedTensorWrapper(x))

        batch = self._native.getNextBatch()
        return {key: from_dlpack(x) for key, x in batch.items()}