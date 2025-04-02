from native_dataloader import Head, Dataset, FileType
import pytest as t
from pathlib import PosixPath

def init_fn():
    pass

def get_dataset(tmp_path: PosixPath, eroneous: bool):
    exts = ["exr", "jpg", "npy"]
    for i, ext in list(zip(range(3), exts))[:2 if eroneous else 3]:
        subdir = tmp_path / f"subdir{i}"
        subdir.mkdir()

        for f in range(10):
            file = subdir / f"file{f}.{ext}"
            file.write_text("Hello World!")

    sh = (100, 100, 3)
    return Dataset.from_subdirs(
        str(tmp_path),
        [
            Head(FileType.EXR, "dictname1", sh),
            Head(FileType.JPG, "dictname2", sh),
            Head(FileType.NPY, "dictname3", sh),
        ],
        ["subdir0", "subdir1", "subdir2"],
        init_fn
    )

def test_get_eroneous_dataset(tmp_path):
    ds = get_dataset(tmp_path, eroneous=True)
    assert len(ds.entries) == 0

def test_get_dataset(tmp_path):
    ds = get_dataset(tmp_path, eroneous=False)
    assert len(ds.entries) == 10
    assert all(len(entry) == 3 for entry in ds.entries)

def verify_dataset_reconstruction(tmp_path, add_trailing_slash: bool, use_absolute_paths: bool):
    ds = get_dataset(tmp_path, eroneous=False)
    sh = (100, 100, 3)
    root_dir = str(tmp_path)

    if use_absolute_paths:
        entries = [[f"{root_dir}{sub_path}" for sub_path in item] for item in ds.entries]
    else:
        entries = ds.entries

    if add_trailing_slash:
        root_dir += '/'

    ds2 = Dataset.from_entries(
        root_dir,
        [
            Head(FileType.EXR, "dictname1", sh),
            Head(FileType.JPG, "dictname2", sh),
            Head(FileType.NPY, "dictname3", sh),
        ],
        entries
    )

    b1 = ds.get_next_batch(len(ds))
    b2 = ds2.get_next_batch(len(ds2))

    assert sorted(tuple(b1)) == sorted(tuple(b2))

def test_dataset_from_entries_normal(tmp_path):
    verify_dataset_reconstruction(tmp_path, False, False)

def test_dataset_from_entries_prepend(tmp_path):
    verify_dataset_reconstruction(tmp_path, False, True)

def test_dataset_from_entries_append(tmp_path):
    verify_dataset_reconstruction(tmp_path, True, False)

def test_dataset_from_entries_prepend_append(tmp_path):
    verify_dataset_reconstruction(tmp_path, True, True)

def test_split_dataset(tmp_path):
    ds = get_dataset(tmp_path, eroneous=False)
    train_ds, valid_ds, test_ds = ds.split_train_validation_test(0.5, 0.2)

    b = ds.get_next_batch(len(ds))
    train_b = train_ds.get_next_batch(len(train_ds))
    valid_b = valid_ds.get_next_batch(len(valid_ds))
    test_b = test_ds.get_next_batch(len(test_ds))

    assert len(train_ds) == 5 and len(valid_ds) == 2 and len(test_ds) == 3
    assert sorted(tuple([*train_b, *valid_b, *test_b])) == sorted(tuple(b))