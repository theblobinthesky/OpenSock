import os
import requests
import tarfile
import io
from glob import glob
import native_dataloader as m
import pytest as pt

JPG_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
JPG_DATASET_DIR = os.path.join("temp", "jpg_dataset")
HEIGHT, WIDTH = 600, 600

def ensure_jpg_dataset():
    # Create the persistent directory if it doesn't exist.
    os.makedirs(JPG_DATASET_DIR, exist_ok=True)
    
    # Check if the directory is empty
    if not os.listdir(JPG_DATASET_DIR):
        response = requests.get(JPG_DATASET_URL)
        response.raise_for_status()
        # Extract the dataset into JPG_DATASET_DIR
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=JPG_DATASET_DIR, filter=lambda tarinfo, _: tarinfo)

        for path in glob('temp/jpg_dataset/flower_photos/*/*.jpg'):
            name = os.path.basename(path)
            os.rename(path, f"temp/jpg_dataset/flower_photos/{name}.jpg")
        
    return "temp/jpg_dataset", "flower_photos"

def get_dataloader(batch_size: int):
    def init_ds_fn():
        pass

    root_dir, sub_dir = ensure_jpg_dataset()
    ds = m.Dataset(root_dir, [m.Subdirectory(sub_dir, m.FileType.JPG, "img", 300, 300)])
    dl = m.DataLoader(ds, batch_size, HEIGHT, WIDTH, init_ds_fn)
    return dl

def test_get_length():
    bs = 16
    dl = get_dataloader(batch_size=bs)
    assert len(dl) == (633 + bs - 1) // bs

# def test_perf():
#     def performance_benchmark(dl: m.DataLoader):
#         for _ in range(len(dl)):
#             batch = dl.getNextBatch()
#             batch = {key: value.shape for key, value in batch.items()}
#             print(batch)

#     dl = get_dataloader(batch_size=16)

#     performance_benchmark(dl)
#     # benchmark(performance_benchmark, dl)
#     assert False