from __future__ import annotations
from ._core import __doc__, __version__
from .native_api import DType, ItemType, Dataset, BatchedDataset, IDataLoader
from .native_api import Codec, CompressorOptions, Compressor, Decompressor
from .native_api import set_device_for_resource_pool, shutdown_resource_pool, decompress_path
from ._core import (
    FlatDataSource,
    FlipAugmentation,
    PadAugmentation,
    PadSettings,
    RandomCropAugmentation,
    ResizeAugmentation,
    DataAugmentationPipe
)

__all__ = [
    "__doc__",
    "__version__",
    "DType",
    "ItemType",
    "Dataset",
    "BatchedDataset",
    "IDataLoader",
    "Codec",
    "CompressorOptions",
    "Compressor",
    "Decompressor",
    "FlatDataSource",
    "FlipAugmentation",
    "PadAugmentation",
    "PadSettings",
    "RandomCropAugmentation",
    "ResizeAugmentation",
    "DataAugmentationPipe",
    "set_device_for_resource_pool",
    "shutdown_resource_pool",
    "pytorch_binding",
    "jax_binding",
    "decompress_path",
]

def __getattr__(name):
    if name in ["pytorch_binding", "jax_binding"]:
        import importlib
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
