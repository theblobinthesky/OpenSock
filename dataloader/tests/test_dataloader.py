import os
import requests
import tarfile
import io
from PIL import Image
import numpy as np
import native_dataloader as m
import pytest as pt
import jax.numpy as jnp

JPG_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
JPG_DATASET_DIR = os.path.join("temp", "jpg_dataset")
HEIGHT, WIDTH = 300, 300
NUM_THREADS, PREFETCH_SIZE = 8, 8

def ensure_jpg_dataset():
    os.makedirs(JPG_DATASET_DIR, exist_ok=True)
    if not os.listdir(JPG_DATASET_DIR):
        response = requests.get(JPG_DATASET_URL)
        response.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=JPG_DATASET_DIR, filter=lambda tarinfo, _: tarinfo)
        
    return "temp/jpg_dataset/flower_photos", "daisy"

def get_dataloader(batch_size: int):
    def init_ds_fn():
        pass

    root_dir, sub_dir = ensure_jpg_dataset()
    ds = m.Dataset(root_dir, [m.Subdirectory(sub_dir, m.FileType.JPG, "img", HEIGHT, WIDTH)])
    dl = m.DataLoader(ds, batch_size, init_ds_fn, NUM_THREADS, PREFETCH_SIZE)
    return ds, dl

def test_get_length():
    bs = 16
    _, dl = get_dataloader(batch_size=bs)
    assert len(dl) == (633 + bs - 1) // bs

def test_correctness():
    bs = 16
    ds, dl = get_dataloader(batch_size=bs)
    ds.init() # TODO: Fix copy bug.
    dsPaths = ds.get_dataset()

    batch = dl.get_next_batch()
    for i, batch_paths in enumerate(dsPaths[:16]):
        path = batch_paths[0]
        pil_img = Image.open(path).convert("RGB")
        pil_img = np.array(pil_img.resize((WIDTH, HEIGHT)), np.float32)
        pil_img = pil_img / 255.0

        err = np.mean(np.abs(batch['img'][i] - pil_img))
        assert np.all(err < 5 / 255.0), f"Error too high for image {path}"

def test_two_dataloaders_simultaneously():
    bs = 16
    _, dl = get_dataloader(batch_size=bs)
    _, dl2 = get_dataloader(batch_size=bs)

    b1 = dl.get_next_batch()
    b2 = dl2.get_next_batch()
    b3 = dl.get_next_batch()

    assert jnp.all(b1['img'] == b2['img']).item()
    assert jnp.all(b1['img'] == b3['img']).item()

def test_perf(benchmark):
    def performance_benchmark(dl: m.DataLoader):
        total_mean = 0.0
        num_iters = len(dl) * 64
        for _ in range(num_iters):
            batch = dl.get_next_batch()
            x = batch['img']
            total_mean += x.mean()

    _, dl = get_dataloader(batch_size=16)
    benchmark(performance_benchmark, dl)
