from . import _core as m
import jax
import jax.numpy as jnp
from .native_api import IDataLoader, Dataset
from typing import Tuple

class JaxSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        # Ensure Jax is using GPU; otherwise raise an error.
        device = jax.devices("gpu")[0]

        def jax_has_gpu():
            try:
                _ = jax.device_put(jax.numpy.ones(1), device=device)
                return True
            except Exception:
                return False

        if jax.default_backend() != "gpu" or not jax_has_gpu():
            raise EnvironmentError("Jax is not using the GPU!")

        x = jnp.zeros((16, 16))
        x = jax.device_put(x, device)
        assert x.mean() == 0.0


class DataLoader(IDataLoader):
    def __init__(
            self,
            dataset: Dataset,
            augPipe,
            batch_size: int,
            num_threads: int = 4,
            prefetch_size: int = 4,
    ):
        """Wrap native DataLoader and ensure Jax GPU is initialized."""
        super().__init__(dataset, augPipe, batch_size, num_threads, prefetch_size)
        JaxSingleton()

    def get_next_batch(self) -> Tuple[dict, dict]:
        """Return next batch as a tuple of (primary_tensors, metadata_tensors).

        primary_tensors: dict of jax arrays with the main data
        metadata_tensors: dict of jax arrays with metadata (e.g., original point counts for POINTS type)
        """

        def from_dlpack(x):
            return jax.dlpack.from_dlpack(x, copy=False)

        batch, metadata = self._native.getNextBatch()
        batch = {key: from_dlpack(x) for key, x in batch.items()}
        metadata = {key: from_dlpack(x) for key, x in metadata.items()}

        # Normalize uint8 inputs to float32 in [0,1]; leave other dtypes intact.
        for key, arr in list(batch.items()):
            if arr.dtype == jnp.uint8:
                batch[key] = arr.astype(jnp.float32) / 255.0

        return batch, metadata
