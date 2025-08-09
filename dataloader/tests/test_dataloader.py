import os
import requests
import tarfile
import io
import native_dataloader as m
import numpy as np
from PIL import Image
import hashlib

JPG_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
JPG_DATASET_DIR = os.path.join("temp", "jpg_dataset")
HEIGHT, WIDTH = 300, 300
NUM_THREADS, PREFETCH_SIZE = 16, 16
NUM_REPEATS = 16

def ensure_jpg_dataset():
    os.makedirs(JPG_DATASET_DIR, exist_ok=True)
    if not os.listdir(JPG_DATASET_DIR):
        response = requests.get(JPG_DATASET_URL)
        response.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=JPG_DATASET_DIR, filter=lambda tarinfo, _: tarinfo)
        
    return "temp/jpg_dataset/flower_photos", "daisy"

def init_ds_fn():
    pass

def get_dataset():
    root_dir, sub_dir = ensure_jpg_dataset()
    return m.Dataset.from_subdirs(root_dir, [m.Head(m.FileType.JPG, "img", (HEIGHT, WIDTH, 3))], [sub_dir], init_ds_fn), root_dir

def get_dataloader(batch_size: int):
    ds, root_dir = get_dataset()
    dl = m.DataLoader(ds, batch_size, NUM_THREADS, PREFETCH_SIZE)
    return ds, dl, root_dir

def get_dataloaders(batch_size: int):
    ds, root_dir = get_dataset()
    ds1, ds2, ds3 = ds.split_train_validation_test(0.3, 0.3)
    return (
        m.DataLoader(ds1, batch_size, NUM_THREADS, PREFETCH_SIZE),
        m.DataLoader(ds2, batch_size, NUM_THREADS, PREFETCH_SIZE),
        m.DataLoader(ds3, batch_size, NUM_THREADS, PREFETCH_SIZE)
    ), (ds1, ds2, ds3), root_dir


def test_get_length():
    bs = 16
    _, dl, _ = get_dataloader(batch_size=bs)
    assert len(dl) == (633 + bs - 1) // bs

import matplotlib.pyplot as plt

def verify_correctness(ds, dl, root_dir, bs):
    entries = [[f"{root_dir}{sub_path}" for sub_path in item] for item in ds.entries]

    for batch_idx in range(len(dl)):
        batch = dl.get_next_batch()
        for i, batch_paths in enumerate(entries[batch_idx * bs:bs + batch_idx * bs]):
            path = batch_paths[0]
            pil_img = Image.open(path).convert("RGB")
            pil_img = np.array(pil_img.resize((WIDTH, HEIGHT)), np.float32)
            pil_img = pil_img / 255.0

            err = np.mean(np.abs(batch['img'][i] - pil_img))
            if not np.all(err < 10 / 255.0):
                print(f"{batch_idx=}, {i=}")
                _, axs = plt.subplots(2, 2)
                axs[0][0].imshow(batch['img'][i])
                axs[0][1].imshow(pil_img)
                plt.show()

            # err = np.mean(np.abs(batch['img'][i] - pil_img))
            assert np.all(err < 10 / 255.0), f"Error too high for image {path}"


def test_one_dataloader_once():
    ds, dl, root_dir = get_dataloader(batch_size=16)
    verify_correctness(ds, dl, root_dir, bs=16)

# def test_one_dataloader_twice():
#     ds, dl, root_dir = get_dataloader(batch_size=16)
#     verify_correctness(ds, dl, root_dir, bs=16)
#     verify_correctness(ds, dl, root_dir, bs=16)

def test_multiple_dlers_without_next_batch():
    (dl1, dl2, dl3), (ds1, ds2, ds3), root_dir = get_dataloaders(batch_size=16)
    verify_correctness(ds1, dl1, root_dir, bs=16)
    verify_correctness(ds2, dl2, root_dir, bs=16)
    verify_correctness(ds3, dl3, root_dir, bs=16)

# def test_correctness_multiple_dlers_with_next_batch():
#     (dl1, dl2, dl3), (ds1, ds2, ds3), root_dir = get_dataloaders(batch_size=16)
#     dl2.get_next_batch()
#     dl1.get_next_batch()
#     dl3.get_next_batch()
#     verify_correctness(ds1, dl1, root_dir, bs=16)
#     print("dl1 ok")
#     verify_correctness(ds2, dl2, root_dir, bs=16)
#     print("dl2 ok")
#     verify_correctness(ds3, dl3, root_dir, bs=16)
#     print("dl3 ok")

# def hash_array(arr):
#     arr_np = np.asarray(arr)
#     return hashlib.md5(arr_np.tobytes()).hexdigest()

# def test_no_duplicates_within_jpg_dataloaders():
#     for i in range(NUM_REPEATS):
#         dl1, dl2, dl3 = get_dataloaders(batch_size=16)
#         dataloaders = {"dl1": dl1, "dl2": dl2, "dl3": dl3}
        
#         for key, dl in dataloaders.items():
#             seen_hashes = set()
#             for _ in range(len(dl)):
#                 batch = dl.get_next_batch()
#                 h = hash_array(batch['img'])
#                 assert h not in seen_hashes, f"Duplicate found within {key}"
#                 seen_hashes.add(h)

# def test_no_duplicates_across_jpg_dataloaders():
#     dl1, dl2, dl3 = get_dataloaders(batch_size=16)
#     dataloaders = {"dl1": dl1, "dl2": dl2, "dl3": dl3}
    
#     overall_hashes = {}
#     for key, dl in dataloaders.items():
#         for _ in range(len(dl)):
#             batch = dl.get_next_batch()
#             for image in batch['img']:
#                 h = hash_array(image)
#                 assert h not in overall_hashes, f"Duplicate found between {overall_hashes[h]} and {key}"
#                 overall_hashes[h] = key

# def test_visualize_batches_in_rows():
#     import matplotlib.pyplot as plt
#     import numpy as np

#     (dl1, dl2, dl3), _, _ = get_dataloaders(batch_size=16)
#     dataloaders = {"dl1": dl1, "dl2": dl2, "dl3": dl3}

#     for name, dl in dataloaders.items():
#         batch_list = []
#         for batch_index in range(len(dl)):
#             batch = dl.get_next_batch()
#             images = batch['img']
#             titles = [f"{batch_index}:{img_index}" for img_index in range(len(images))]
#             batch_list.append((images, titles))
            
#         nrows = len(batch_list)
#         ncols = max(len(images) for images, _ in batch_list)

#         _, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
#         if nrows == 1:
#             axs = np.expand_dims(axs, axis=0)
#         if ncols == 1:
#             axs = np.expand_dims(axs, axis=1)
        
#         for row_idx, (images, titles) in enumerate(batch_list):
#             for col_idx in range(ncols):
#                 ax = axs[row_idx][col_idx]
#                 if col_idx < len(images):
#                     ax.imshow(images[col_idx])
#                     ax.set_title(titles[col_idx], fontsize=8)
#                     ax.axis('off')
#                 else:
#                     ax.set_visible(False)

#         plt.suptitle(name)
#         plt.tight_layout()
#         plt.show()


# def test_visualize_3_dataloaders_three_rows_each(batch_size=16):
#     import matplotlib.pyplot as plt
#     import numpy as np

#     (dl1, dl2, dl3), _, _ = get_dataloaders(batch_size=batch_size)
#     dl2.get_next_batch()
#     dataloaders = {"dl1": dl1, "dl2": dl2, "dl3": dl3}
#     rows = 3

#     plot_batches = []  # Each element: (loader_name, images, titles)
#     max_imgs_in_row = 0

#     for loader_name, dl in dataloaders.items():
#         for batch_idx in range(len(dl)):
#             batch = dl.get_next_batch()
#             imgs = batch["img"]
#             titles = [f"{loader_name} {batch_idx}:{i}" for i in range(len(imgs))]
#             if batch_idx < rows:
#                 plot_batches.append((loader_name, imgs, titles))
#             max_imgs_in_row = max(max_imgs_in_row, len(imgs))

#     total_rows = len(dataloaders) * rows
#     total_cols = max_imgs_in_row

#     _, axs = plt.subplots(total_rows, total_cols, figsize=(total_cols * 2, total_rows * 2))

#     if total_rows == 1:
#         axs = np.expand_dims(axs, axis=0)
#     if total_cols == 1:
#         axs = np.expand_dims(axs, axis=1)

#     for row_idx, (loader_name, imgs, titles) in enumerate(plot_batches):
#         for col_idx in range(total_cols):
#             ax = axs[row_idx][col_idx]
#             if col_idx < len(imgs):
#                 ax.imshow(imgs[col_idx])
#                 ax.set_title(titles[col_idx], fontsize=7)
#                 ax.axis("off")
#             else:
#                 ax.set_visible(False)

#     plt.suptitle(f"First {rows} batches (rows) for each dataloader "
#                  f"— columns vary with batch size", fontsize=14, y=1.02)
#     plt.tight_layout()
#     plt.show()


# def test_perf(benchmark):
#     def performance_benchmark(dl: m.DataLoader):
#         total_mean = 0.0
#         num_iters = len(dl) * 64
#         for _ in range(num_iters):
#             batch = dl.get_next_batch()
#             x = batch['img']
#             total_mean += x.mean()

#     _, dl, _ = get_dataloader(batch_size=16)
#     benchmark(performance_benchmark, dl)

# def test_correctness_npy(tmp_path):
#     subdir = tmp_path / "subdir"
#     subdir.mkdir()

#     for i in range(16):
#         testFile = tmp_path / "subdir" / f"file{i}"
#         np.save(testFile, np.ones((1, 3, 3, 4), dtype=np.float32))

#     ds = m.Dataset.from_subdirs(
#         str(tmp_path), 
#         [m.Head(m.FileType.NPY, "np", (3, 3, 4))],
#         ["subdir"],
#         init_ds_fn
#     )

#     dl = m.DataLoader(ds, 16, NUM_THREADS, PREFETCH_SIZE)

#     batch = dl.get_next_batch()['np']
#     assert np.all(np.ones((16, 3, 3, 4)) == batch)

# def test_two_dataloaders_simultaneously():
#     bs = 16
#     _, dl, _ = get_dataloader(batch_size=bs)
#     _, dl2, _ = get_dataloader(batch_size=bs)

#     b1 = dl.get_next_batch()
#     b2 = dl2.get_next_batch()
#     b3 = dl.get_next_batch()

#     assert jnp.all(b1['img'] == b2['img']).item()
#     assert jnp.all(b1['img'] == b3['img']).item()
