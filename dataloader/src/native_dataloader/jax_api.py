from enum import Enum
from typing import Callable, List
import jax, jax.numpy as jnp
from . import _core as m  # Import native module as m

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

class FileType(Enum):
    EXR = m.FileType.EXR
    JPG = m.FileType.JPG
    NPY = m.FileType.NPY

class Subdirectory:
    def __init__(self, path: str, file_type: FileType, dict_name: str,
                 image_width: int, image_height: int):
        """Wrap native Subdirectory."""
        self._native = m.Subdirectory(path, file_type.value, dict_name,
                                      image_width, image_height)

    def native(self) -> m.Subdirectory:
        """Return native Subdirectory."""
        return self._native

class Dataset:
    def __init__(self, root_dir: str, sub_dirs: List[Subdirectory]):
        """Wrap native Dataset using a list of native Subdirectories."""
        self._native = m.Dataset(root_dir, [s.native() for s in sub_dirs])

    def init(self) -> None:
        """Initialize the dataset."""
        self._native.init()

    def get_dataset(self):
        """Return dataset samples."""
        return self._native.getDataset()

    def native(self) -> m.Dataset:
        """Return native Dataset."""
        return self._native

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._native)

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int,
                 init_fn: Callable[[], None],
                 num_threads: int = 4,
                 prefetch_size: int = 4):
        """Wrap native DataLoader and ensure Jax GPU is initialized."""
        JaxSingleton()
        self._native = m.DataLoader(dataset.native(), batch_size, init_fn,
                                    num_threads, prefetch_size)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._native)

    def get_next_batch(self) -> dict:
        """Return next batch as a dict of jax arrays."""
        def from_dlpack(x):
            return jax.dlpack.from_dlpack(DLManagedTensorWrapper(x))
        batch = self._native.getNextBatch()
        return {key: from_dlpack(x) for key, x in batch.items()}
