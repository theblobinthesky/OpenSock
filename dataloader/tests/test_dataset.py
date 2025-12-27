from pathlib import PosixPath
import pytest as t
from native_dataloader import Dataset, BatchedDataset, DataLoader
import numpy as np
from PIL import Image as pimg


def init_fn():
    pass

def make_dataset(tmp_path: PosixPath, erroneous: bool, add_trailing_slash: bool=False) -> Dataset:
    exts = ["png", "jpg", "npy"]
    mapping = [(f"subdir{i}", f"dictname{i + 1}") for i in range(3)]

    for i, ext in list(zip(range(3), exts))[:2 if erroneous else 3]:
        subdir = tmp_path / f"subdir{i}"
        subdir.mkdir()
        for f in range(10):
            file = subdir / f"file{f}.{ext}"

            if ext == "png" or ext == "jpg":
                pimg.new('RGB', (512, 512)).save(file)
            elif ext == "npy":
                np.save(file, np.zeros((2, 2), np.float32))
            else:
                raise ValueError("Unsupported file extension.")

    root_dir = str(tmp_path)
    if add_trailing_slash:
        root_dir += "/"

    return Dataset.from_subdirs(root_dir, mapping, init_fn)


def get_batched_dataset(ds: Dataset):
    return BatchedDataset(ds, len(ds))


def test_get_eroneous_dataset(tmp_path):
    with t.raises(RuntimeError):
        make_dataset(tmp_path, erroneous=True)


def test_get_dataset(tmp_path):
    ds = make_dataset(tmp_path, erroneous=False)
    assert len(ds.entries) == 10
    assert all(len(entry) == 3 for entry in ds.entries)


def test_dataset_works_with_trailing_slash(tmp_path):
    ds = make_dataset(tmp_path, erroneous=False)
    assert len(ds.entries) == 10

def test_split_dataset(tmp_path):
    ds = make_dataset(tmp_path, erroneous=False)
    assert len(ds) == 10

    train_ds, valid_ds, test_ds = ds.split_train_validation_test(0.5, 0.2)

    b = get_batched_dataset(ds).get_next_batch()
    train_b = get_batched_dataset(train_ds).get_next_batch()
    valid_b = get_batched_dataset(valid_ds).get_next_batch()
    test_b = get_batched_dataset(test_ds).get_next_batch()

    assert len(train_ds) == 5 and len(valid_ds) == 2 and len(test_ds) == 3
    assert sorted(tuple([*train_b, *valid_b, *test_b])) == sorted(tuple(b))
