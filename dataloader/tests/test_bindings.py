import pytest
import native_dataloader as m
from native_dataloader.jax_binding import DataLoader as JaxDataLoader
from native_dataloader.pytorch_binding import DataLoader as PyTorchDataLoader
import numpy as np

BATCH_SIZE = 16
NUM_THREADS = 8
PREFETCH_SIZE = 4

@pytest.fixture(params=[
    ("jax", JaxDataLoader),
    ("pytorch", PyTorchDataLoader),
], ids=lambda p: p[0]")
def loader_cfg(request):
    name, cls = request.param
    return {"name": name, "cls": cls}

def init_ds_fn():
    pass

def get_empty_pipe():
    return m.DataAugmentationPipe([m.PadAugmentation(1024, 1024, m.PadSettings.PAD_BOTTOM_RIGHT)], [16, 1024, 1024, 3], 128, 4)

def test_binding(tmp_path, loader_cfg):
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    for i in range(16):
        testFile = tmp_path / "subdir" / f"file{i}"
        np.save(testFile, np.ones((3, 3, 4), dtype=np.float32))

    ds = m.Dataset.from_subdirs(str(tmp_path), [("subdir", "np", m.ItemType.NONE)], init_ds_fn)

    cls = loader_cfg['cls']
    dl = cls(ds, 16, NUM_THREADS, PREFETCH_SIZE, get_empty_pipe())

    batch, _ = dl.get_next_batch()
    np_data = batch["np"]
    assert np.all(np.ones((16, 3, 3, 4)) == np_data)

    return ds, dl
