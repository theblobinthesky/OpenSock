import os
import requests
import tarfile
import io
from PIL import Image
import numpy as np
import jax, jax.numpy as jnp
import time
import native_dataloader as m
import pytest as pt

from jax._src.lib import xla_client as xc
from jax.core import ShapedArray
from jax.sharding import SingleDeviceSharding

JPG_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
JPG_DATASET_DIR = os.path.join("temp", "jpg_dataset")
HEIGHT, WIDTH = 300, 300
NUM_THREADS, PREFETCH_SIZE = 16, 16

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
    dsPaths = ds.getDataset()

    batch = dl.getNextBatch()
    for i, batch_paths in enumerate(dsPaths[:16]):
        path = batch_paths[0]
        pil_img = Image.open(path).convert("RGB")
        pil_img = np.array(pil_img.resize((WIDTH, HEIGHT)), np.float32)
        native_img = batch["img"][i]
        native_img = native_img.astype(np.float32)

        err = np.mean(np.abs(native_img - pil_img))
        assert np.all(err < 5), f"Error too high for image {path}"

def test_perf(benchmark):
    def jax_has_gpu():
        try:
            _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
            return True
        except:
            return False
    assert jax.default_backend() == 'gpu' and jax_has_gpu() 

    def fast_numpy_to_device_array(x):
        aval = ShapedArray(x.shape, x.dtype)
        return xc.batched_device_put(aval, sharding, [x], [device], True)

    def performance_benchmark(dl: m.DataLoader):
        total_mean = 0.0
        total_dur = 0.0
        num_iters = len(dl) * 64
        for _ in range(num_iters):
            batch = dl.getNextBatch()

            s = time.time()
            x = batch["img"]
            x = jax.device_put(x, device) # , jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host"))
            jax.block_until_ready(x)
            dur = time.time() - s
            total_dur += dur

            total_mean += x.mean()

        dur = total_dur / num_iters

        print()
        print(f"Time taken: {dur}; GiB/s: {BATCH_SIZE_IN_GB/dur}, total_mean={total_mean}")


    _, dl = get_dataloader(batch_size=256)
    BATCH_SIZE_IN_GB = dl.getNextBatch()["img"].size / 1024 / 1024 / 1024

    device = jax.devices()[0]
    sharding = SingleDeviceSharding(device)

    benchmark(performance_benchmark, dl)
