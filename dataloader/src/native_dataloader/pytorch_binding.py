from . import _core as m
import torch
import torch.utils.dlpack
from .native_api import IDataLoader, Dataset
from typing import Tuple


class TorchSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        # Ensure Torch is using GPU; otherwise raise an error.
        if not torch.cuda.is_available():
            raise EnvironmentError("Torch is not using the GPU!")

        try:
            # Verify we can allocate and operate on the device
            device = torch.device("cuda")
            x = torch.zeros((16, 16), device=device)
            assert x.mean().item() == 0.0
        except Exception as e:
            raise EnvironmentError(f"Failed to initialize Torch on GPU: {e}")


class DataLoader(IDataLoader):
    def __init__(
            self,
            dataset: Dataset,
            augPipe,
            batch_size: int,
            num_threads: int = 4,
            prefetch_size: int = 4,
    ):
        """Wrap native DataLoader and ensure Torch GPU is initialized."""
        super().__init__(dataset, augPipe, batch_size, num_threads, prefetch_size)
        TorchSingleton()

    def get_next_batch(self) -> Tuple[dict, dict]:
        """Return next batch as a tuple of (primary_tensors, metadata_tensors).

        primary_tensors: dict of torch tensors with the main data
        metadata_tensors: dict of torch tensors with metadata
        """

        def from_dlpack(x):
            return torch.from_dlpack(x)

        batch, metadata = self._native.getNextBatch()
        batch = {key: from_dlpack(x) for key, x in batch.items()}
        metadata = {key: from_dlpack(x) for key, x in metadata.items()}

        # Normalize uint8 inputs to float32 in [0,1]; leave other dtypes intact.
        for key, arr in list(batch.items()):
            if arr.dtype == torch.uint8:
                batch[key] = arr.to(dtype=torch.float32) / 255.0

        return batch, metadata
