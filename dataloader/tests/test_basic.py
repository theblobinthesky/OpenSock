from __future__ import annotations

import native_dataloader as m
import pytest as t


def test_version():
    assert m.__version__ == "0.0.1"

def test_wrong_constructor_args():
    with t.raises(TypeError):
        ds = m.Dataset(1)

def init_ds_func():
    print("Hello World!")

def get_dataset():
    ds = m.Dataset(
        "testdir",
        [
            m.Subdirectory("subdir1", m.FileType.EXR, "dictname1"),
            m.Subdirectory("subdir2", m.FileType.JPG, "dictname2"),
            m.Subdirectory("subdir2", m.FileType.NPY, "dictname2"),
        ],
        init_ds_func
    )

    return ds


def test_constructor():
    ds = get_dataset()
    ds.init()
    assert False
