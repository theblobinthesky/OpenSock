import os
import pytest
import requests
import tarfile
import io
import hashlib
import native_dataloader as m
import numpy as np
from PIL import Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 # for exr

JPG_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
JPG_DATASET_DIR = os.path.join("temp", "jpg_dataset")

PNG_DATASET_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/PNG-Gradient_hex.png/250px-PNG-Gradient_hex.png"
PNG_DATASET_FILE = os.path.join("temp", "img", "file.png")

EXR_DATASET_URL = "https://github.com/ampas/ACES_ODT_SampleFrames/raw/refs/heads/main/ACES_OT_VWG_SampleFrames/ACES_OT_VWG_SampleFrames.0001.exr"
EXR_DATASET_FILE = os.path.join("temp", "img", "file.exr")

HEIGHT, WIDTH = 300, 300

# List of (num_threads, prefetch_size) configurations to test.
# Edit this list to try different combinations.
DL_CONFIGS = [
    # (NUM_THREADS, PREFETCH_SIZE)
    (16, 16),
    (16, 4),
    (8, 16),
    (8, 1),
    (8, 2)
]
NUM_THREADS, PREFETCH_SIZE = 16, 4

# Pytest fixture to parameterize tests over thread/prefetch configurations.
@pytest.fixture(params=DL_CONFIGS, ids=lambda p: f"threads={p[0]},prefetch={p[1]}")
def dl_cfg(request):
    num_threads, prefetch_size = request.param
    return {"num_threads": num_threads, "prefetch_size": prefetch_size}
NUM_REPEATS = 16

def ensure_jpg_dataset():
    os.makedirs(JPG_DATASET_DIR, exist_ok=True)
    if not os.listdir(JPG_DATASET_DIR):
        response = requests.get(JPG_DATASET_URL)
        response.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=JPG_DATASET_DIR, filter=lambda tarinfo, _: tarinfo)
        
    return "temp/jpg_dataset/flower_photos", "daisy"

def download_file(url: str, path: str):
    os.makedirs("temp/img", exist_ok=True)
    if not os.path.exists(path):
        response = requests.get(url)
        response.raise_for_status()
        with open(path, "wb") as file:
            bytes = io.BytesIO(response.content).read()
            file.write(bytes)


def init_ds_fn():
    pass

def get_dataset():
    root_dir, sub_dir = ensure_jpg_dataset()
    return m.Dataset.from_subdirs(root_dir, [m.Head(m.FileType.JPG, "img", (HEIGHT, WIDTH, 3))], [sub_dir], init_ds_fn), root_dir

def get_dataloader(batch_size: int, num_threads: int, prefetch_size: int):
    ds, root_dir = get_dataset()
    dl = m.DataLoader(ds, batch_size, num_threads, prefetch_size)
    return ds, dl, root_dir

def get_dataloaders(batch_size: int, num_threads: int, prefetch_size: int):
    ds, root_dir = get_dataset()
    ds1, ds2, ds3 = ds.split_train_validation_test(0.3, 0.3)
    return (
        m.DataLoader(ds1, batch_size, num_threads, prefetch_size),
        m.DataLoader(ds2, batch_size, num_threads, prefetch_size),
        m.DataLoader(ds3, batch_size, num_threads, prefetch_size)
    ), (ds1, ds2, ds3), root_dir


def test_get_length(dl_cfg):
    bs = 16
    _, dl, _ = get_dataloader(batch_size=bs, **dl_cfg)
    assert len(dl) == (633 + bs - 1) // bs


def verify_correctness(ds, dl, root_dir, bs, reps=1, start=0):
    # Continuous wrapping across all batches and repetitions (no reset between reps)
    total_batches = len(dl) * reps
    paths = [f"{root_dir}{item[0]}" for item in ds.entries]

    for global_batch_idx in range(total_batches):
        batch = dl.get_next_batch()
        imgs = batch['img']

        for i in range(bs):
            path = paths[start % len(ds)]
            pil_img = Image.open(path).convert("RGB")
            pil_img = np.array(pil_img.resize((WIDTH, HEIGHT)), np.float32) / 255.0

            err = np.mean(np.abs(imgs[i] - pil_img))
            if not np.all(err < 10 / 255.0):
                print(f"not matching on {global_batch_idx=}, {i=}")
                import matplotlib.pyplot as plt
                _, axs = plt.subplots(2, 2)
                axs[0][0].imshow(imgs[i])
                axs[0][1].imshow(pil_img)
                plt.show()

            assert np.all(err < 10 / 255.0), f"Error too high for image {path}"
            start += 1


def test_one_dataloader_once(dl_cfg):
    ds, dl, root_dir = get_dataloader(batch_size=16, **dl_cfg)
    verify_correctness(ds, dl, root_dir, bs=16)

def test_one_dataloader_trice(dl_cfg):
    ds, dl, root_dir = get_dataloader(batch_size=16, **dl_cfg)
    verify_correctness(ds, dl, root_dir, bs=16, reps=3)

def test_three_dlers_without_next_batch(dl_cfg):
    (dl1, dl2, dl3), (ds1, ds2, ds3), root_dir = get_dataloaders(batch_size=16, **dl_cfg)
    verify_correctness(ds1, dl1, root_dir, bs=16)
    verify_correctness(ds2, dl2, root_dir, bs=16)
    verify_correctness(ds3, dl3, root_dir, bs=16)

def test_three_dlers_with_next_batch(dl_cfg):
    bs = 16
    (dl1, dl2, dl3), (ds1, ds2, ds3), root_dir = get_dataloaders(batch_size=bs, **dl_cfg)
    dl2.get_next_batch()
    dl1.get_next_batch()
    dl3.get_next_batch()
    verify_correctness(ds1, dl1, root_dir, bs=bs, start=bs)
    verify_correctness(ds2, dl2, root_dir, bs=bs, start=bs)
    verify_correctness(ds3, dl3, root_dir, bs=bs, start=bs)

def test_two_dlers_with_different_batch_sizes(dl_cfg):
    ds, dl, root_dir = get_dataloader(batch_size=16, **dl_cfg)
    ds2, dl2, root_dir = get_dataloader(batch_size=9, **dl_cfg)
    verify_correctness(ds, dl, root_dir, bs=16)
    verify_correctness(ds, dl2, root_dir, bs=9)

def hash_array(tensor_like) -> str:
    arr = np.ascontiguousarray(np.asarray(tensor_like))
    meta = f"{arr.dtype}|{arr.shape}|".encode()
    return hashlib.sha256(meta + arr.tobytes()).hexdigest()

def iter_images_once(dl, n):
    yielded = 0
    for _ in range(len(dl)):
        batch = dl.get_next_batch()
        for img in batch['img']:
            if yielded >= n:
                return
            yield img
            yielded += 1

def collect_hashes_once(dl, ds):
    return [hash_array(img) for img in iter_images_once(dl, len(ds))]

def test_no_duplicates_within_jpg_dataloaders(dl_cfg):
    (dl1, dl2, dl3), (ds1, ds2, ds3), _ = get_dataloaders(batch_size=16, **dl_cfg)
    for name, dl, ds in zip(("train","validation","test"), (dl1,dl2,dl3), (ds1,ds2,ds3)):
        hashes = collect_hashes_once(dl, ds)
        assert len(hashes) == len(ds)
        assert len(set(hashes)) == len(ds), f"Duplicate hash found within '{name}' dataloader."

def test_no_duplicates_across_jpg_dataloaders(dl_cfg):
    (dl1, dl2, dl3), (ds1, ds2, ds3), _ = get_dataloaders(batch_size=16, **dl_cfg)
    names = ("train","validation","test")
    overall = {}
    for name, dl, ds in zip(names, (dl1,dl2,dl3), (ds1,ds2,ds3)):
        for h in collect_hashes_once(dl, ds):
            if h in overall:
                raise AssertionError(
                    f"Duplicate found! Image in '{name}' was already in '{overall[h]}'."
                )
            overall[h] = name
    total = len(ds1) + len(ds2) + len(ds3)
    assert len(overall) == total


def test_correctness_exr():
    download_file(EXR_DATASET_URL, EXR_DATASET_FILE)
    ds = m.Dataset.from_subdirs("temp", [m.Head(m.FileType.EXR, "img", (1080, 1920, 3))], ["img"], init_ds_fn)
    dl = m.DataLoader(ds, 1, NUM_THREADS, PREFETCH_SIZE)
    img = dl.get_next_batch()['img'][0]

    gt_img = cv2.imread(EXR_DATASET_FILE, cv2.IMREAD_UNCHANGED).astype(np.float32)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    assert np.allclose(img, gt_img)

def test_correctness_png():
    download_file(PNG_DATASET_URL, PNG_DATASET_FILE)
    ds = m.Dataset.from_subdirs("temp", [m.Head(m.FileType.PNG, "img", (191, 250, 3))], ["img"], init_ds_fn)
    dl = m.DataLoader(ds, 1, NUM_THREADS, PREFETCH_SIZE)
    img = dl.get_next_batch()['img'][0]

    pil_img = np.array(Image.open(PNG_DATASET_FILE).convert("RGB"), np.float32)
    pil_img = pil_img / 255.0

    assert np.all((img - pil_img) < 1 / 255.0)

def test_correctness_npy(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    for i in range(16):
        testFile = tmp_path / "subdir" / f"file{i}"
        np.save(testFile, np.ones((1, 3, 3, 4), dtype=np.float32))

    ds = m.Dataset.from_subdirs(
        str(tmp_path), 
        [m.Head(m.FileType.NPY, "np", (3, 3, 4))],
        ["subdir"],
        init_ds_fn
    )

    dl = m.DataLoader(ds, 16, NUM_THREADS, PREFETCH_SIZE)

    batch = dl.get_next_batch()['np']
    assert np.all(np.ones((16, 3, 3, 4)) == batch)

@pytest.mark.parametrize("cast_to_fp16", [True, False])
def test_correctness_compressed_trivial(tmp_path, cast_to_fp16):
    def _make_trivial_npy_dir(root: str, subdir: str, n_files: int = 8, shape=(2, 3, 1)) -> str:
        d = os.path.join(root, subdir)
        os.makedirs(d, exist_ok=True)
        H, W, C = shape
        for i in range(n_files):
            arr = np.arange(H * W * C, dtype=np.float32).reshape(H, W, C) + i
            np.save(os.path.join(d, f"file{i}"), arr)
        return d

    # Prepare small npy inputs
    in_sub = _make_trivial_npy_dir(str(tmp_path), "in", n_files=10, shape=(2, 3, 1))
    out_sub = os.path.join(str(tmp_path), "out")
    os.makedirs(out_sub, exist_ok=True)

    # Compress them using the native compressor
    shape = [2, 3, 1]
    options = m.CompressorOptions(
        num_threads=4,
        input_directory=in_sub,
        output_directory=out_sub,
        shape=shape,
        cast_to_fp16=cast_to_fp16,
        permutations=[[0, 1, 2]],
        with_bitshuffle=False,
        allowed_codecs=[],
        tolerance_for_worse_codec=0.01,
    )
    m.Compressor(options).start()

    # Build dataset + dataloader consuming compressed files
    ds = m.Dataset.from_subdirs(
        str(tmp_path),
        [m.Head(m.FileType.COMPRESSED, "feat", shape)],
        ["out"],
        init_ds_fn,
    )
    dl = m.DataLoader(ds, 4, 4, 2)

    # Verify contents for one batch
    batch = dl.get_next_batch()["feat"]
    assert batch.shape[1:] == (2, 3, 1)
    # Expect float32 output from dataloader for now
    assert batch.dtype == np.float32

    # Compare to originals (allow tolerance if fp16)
    tol = 1e-2 if cast_to_fp16 else 0.0
    for i in range(min(4, len(ds))):
        orig = np.load(os.path.join(in_sub, f"file{i}.npy")).astype(np.float32)
        assert np.allclose(batch[i], orig, atol=tol, rtol=0)


def test_end_to_end_perf(benchmark):
    bs = 16
    _, dl, _ = get_dataloader(batch_size=bs, num_threads=24, prefetch_size=3)

    # Warmup a bit to fill prefetch and spin up threads
    for _ in range(min(2, len(dl))):
        _ = dl.get_next_batch()

    num_batches = len(dl) * 16  # keep runtime reasonable

    def fetch_loop():
        items = 0
        for _ in range(num_batches):
            batch = dl.get_next_batch()
            x = batch['img']
            # Touch shape/size to ensure the object is realized and then drop it
            items += int(x.size)
            del batch
        return items

    total_items = benchmark(fetch_loop)
    # Sanity check to keep benchmark from being optimized away
    assert total_items > 0
