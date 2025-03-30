import os
import requests
import tarfile
import io
from PIL import Image
import numpy as np
import jax
import native_dataloader as m
import pytest as pt
import time

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

class DLManagedTensorWrapper:
    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, stream=None):
        return self._capsule

    def __dlpack_device__(self):
        # Return a tuple (device_type, device_id).
        # Assuming CUDA is represented by 2 and you're on GPU 0.
        kDLCUDA = 2
        return (kDLCUDA, 0)

def test_correctness():
    bs = 16
    ds, dl = get_dataloader(batch_size=bs)
    ds.init() # TODO: Fix copy bug.
    dsPaths = ds.getDataset()

    batch = dl.getNextBatch()
    dlPack = DLManagedTensorWrapper(batch["img"])
    native_img = jax.dlpack.from_dlpack(dlPack)
    for i, batch_paths in enumerate(dsPaths[:16]):
        path = batch_paths[0]
        pil_img = Image.open(path).convert("RGB")
        pil_img = np.array(pil_img.resize((WIDTH, HEIGHT)), np.float32)
        pil_img = pil_img / 255.0

        err = np.mean(np.abs(native_img[i] - pil_img))
        assert np.all(err < 5 / 255.0), f"Error too high for image {path}"

# def test_perf(benchmark):
#     def jax_has_gpu():
#         try:
#             _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
#             return True
#         except:
#             return False
#     assert jax.default_backend() == 'gpu' and jax_has_gpu() 

#     def from_dlpack(x):
#         dlPack = DLManagedTensorWrapper(x)
#         return jax.dlpack.from_dlpack(dlPack)

#     def performance_benchmark(dl: m.DataLoader):
#         total_mean = 0.0
#         total_dur = 0.0
#         num_iters = len(dl) * 64
#         for _ in range(num_iters):
#             batch = dl.getNextBatch()

#             s = time.time()
#             x = from_dlpack(batch["img"])
#             jax.block_until_ready(x)
#             dur = time.time() - s
#             total_dur += dur

#             total_mean += x.mean()

#         dur = total_dur / num_iters

#         print()
#         print(f"Time taken: {dur}; GiB/s: {BATCH_SIZE_IN_GB/dur}, total_mean={total_mean}")


#     _, dl = get_dataloader(batch_size=256)
#     BATCH_SIZE_IN_GB = from_dlpack(dl.getNextBatch()["img"]).size * 4 / 1024 / 1024 / 1024
#     benchmark(performance_benchmark, dl)
