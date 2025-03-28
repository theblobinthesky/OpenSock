import native_dataloader as m
import pytest as t
from pathlib import PosixPath

def test_version():
    assert m.__version__ == "0.0.1"

def test_wrong_constructor_args():
    with t.raises(TypeError):
        m.Dataset(1)

def test_wrong_constructor_args():
    with t.raises(RuntimeError):
        m.Dataset("root", [])

def get_dataset(tmp_path: PosixPath, eroneous: bool):
    exts = ["exr", "jpg", "npy"]
    for i, ext in list(zip(range(3), exts))[:2 if eroneous else 3]:
        subdir = tmp_path / f"subdir{i}"
        subdir.mkdir()

        for f in range(4):
            file = subdir / f"file{f}.{ext}"
            file.write_text("Hello World!")

    ds = m.Dataset(
        str(tmp_path),
        [
            m.Subdirectory("subdir0", m.FileType.EXR, "dictname1", 100, 100),
            m.Subdirectory("subdir1", m.FileType.JPG, "dictname2", 100, 100),
            m.Subdirectory("subdir2", m.FileType.NPY, "dictname3", 100, 100),
        ],
    )
    ds.init()
    return ds

def test_get_eroneous_dataset(tmp_path):
    ds = get_dataset(tmp_path, eroneous=True)
    ds = ds.getDataset()
    assert len(ds) == 0

def test_get_dataset(tmp_path):
    ds = get_dataset(tmp_path, eroneous=False)
    ds = ds.getDataset()
    print(ds)
    assert len(ds) == 4
    assert all(len(entry) == 3 for entry in ds)

