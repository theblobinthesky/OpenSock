import os, requests
import io, tarfile, hashlib
import pytest
import native_dataloader as m
from native_dataloader.jax_binding import DataLoader
import numpy as np
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # for exr

JPG_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
JPG_DATASET_DIR = os.path.join("temp", "jpg_dataset")

PNG_DATASET_FILE = os.path.join("temp", "img_png", "file.png")
EXR_DATASET_FILE = os.path.join("temp", "img_exr", "file.exr")

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
DEF_BATCH_SIZE, DEF_NUM_THREADS, DEF_PREFETCH_SIZE = 16, 16, 4


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
    os.makedirs("temp/img_exr", exist_ok=True)
    os.makedirs("temp/img_png", exist_ok=True)
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
    return m.Dataset.from_subdirs(
        root_dir, 
        [(sub_dir, "img", m.ItemType.RASTER)], 
        init_ds_fn
    )

def get_npy_dataset(tmp_path):
    subdir = tmp_path / "npy_subdir"
    subdir.mkdir()

    for i in range(16):
        testFile = tmp_path / "npy_subdir" / f"file{i}"
        np.save(testFile, np.ones((3, 3, 4), dtype=np.float32))

    return m.Dataset.from_subdirs(str(tmp_path), [("npy_subdir", "np", m.ItemType.NONE)], init_ds_fn)


def get_aug_pipe():
    return m.DataAugmentationPipe([m.ResizeAugmentation(HEIGHT, WIDTH)], [16, 500, 700, 3], 128, 4)

def get_pad_pipe():
    return m.DataAugmentationPipe([m.PadAugmentation(1024, 1024, m.PadSettings.PAD_BOTTOM_RIGHT)], [16, 1024, 1024, 3], 128, 4)

def get_dataloader(batch_size: int, num_threads: int, prefetch_size: int):
    ds = get_dataset()
    dl = DataLoader(ds, batch_size, num_threads, prefetch_size, get_aug_pipe())
    return ds, dl


def get_dataloaders(batch_size: int, num_threads: int, prefetch_size: int):
    ds = get_dataset()
    aug_pipe = get_aug_pipe()
    ds1, ds2, ds3 = ds.split_train_validation_test(0.3, 0.3)
    return (
        (
            DataLoader(ds1, batch_size, num_threads, prefetch_size, aug_pipe),
            DataLoader(ds2, batch_size, num_threads, prefetch_size, aug_pipe),
            DataLoader(ds3, batch_size, num_threads, prefetch_size, aug_pipe),
        ),
        (ds1, ds2, ds3)
    )


def assert_low_error(img, gt_img):
    err = np.mean(np.abs(img - gt_img))
    assert np.all(err < 10 / 255.0), f"Error too high for image"

def verify_correctness(ds, dl, bs, reps=1, start=0):
    # Continuous wrapping across all batches and repetitions (no reset between reps)
    total_batches = len(dl) * reps
    paths = [item[0] for item in ds.entries]

    for _ in range(total_batches):
        batch, _ = dl.get_next_batch()
        imgs = batch["img"]

        for i in range(bs):
            path = paths[start % len(ds)]
            pil_img = Image.open(path).convert("RGB")
            pil_img = np.array(pil_img.resize((WIDTH, HEIGHT)), np.float32) / 255.0
            assert_low_error(imgs[i], pil_img)
            start += 1

class TestGeneral:
    def test_get_length(self, dl_cfg):
        bs = 16
        _, dl = get_dataloader(batch_size=bs, **dl_cfg)
        assert len(dl) == (633 + bs - 1) // bs

    def test_item_type_none_shapes_must_be_consistent(self):
        raise ValueError("Test")


class TestRaster:
    def test_one_dataloader_once(self, dl_cfg):
        ds, dl = get_dataloader(batch_size=16, **dl_cfg)
        verify_correctness(ds, dl, bs=16)

    def test_one_dataloader_trice(self, dl_cfg):
        ds, dl = get_dataloader(batch_size=16, **dl_cfg)
        verify_correctness(ds, dl, bs=16, reps=3)

    def test_three_dlers_without_next_batch(self, dl_cfg):
        (dl1, dl2, dl3), (ds1, ds2, ds3) = get_dataloaders(
            batch_size=16, **dl_cfg
        )
        verify_correctness(ds1, dl1, bs=16)
        verify_correctness(ds2, dl2, bs=16)
        verify_correctness(ds3, dl3, bs=16)

    def test_three_dlers_with_next_batch(self, dl_cfg):
        bs = 16
        (dl1, dl2, dl3), (ds1, ds2, ds3) = get_dataloaders(
            batch_size=bs, **dl_cfg
        )
        _, _ = dl2.get_next_batch()
        _, _ = dl1.get_next_batch()
        _, _ = dl3.get_next_batch()
        verify_correctness(ds1, dl1, bs=bs, start=bs)
        verify_correctness(ds2, dl2, bs=bs, start=bs)
        verify_correctness(ds3, dl3, bs=bs, start=bs)

    def test_two_dlers_with_different_batch_sizes(self, dl_cfg):
        ds, dl = get_dataloader(batch_size=16, **dl_cfg)
        ds2, dl2 = get_dataloader(batch_size=9, **dl_cfg)
        verify_correctness(ds, dl, bs=16)
        verify_correctness(ds2, dl2, bs=9)

    def hash_array(self, tensor_like) -> str:
        arr = np.ascontiguousarray(np.asarray(tensor_like))
        meta = f"{arr.dtype}|{arr.shape}|".encode()
        return hashlib.sha256(meta + arr.tobytes()).hexdigest()

    def iter_images_once(self, dl, n):
        yielded = 0
        for _ in range(len(dl)):
            batch, _ = dl.get_next_batch()
            for img in batch["img"]:
                if yielded >= n:
                    return
                yield img
                yielded += 1

    def collect_hashes_once(self, dl, ds):
        return [self.hash_array(img) for img in self.iter_images_once(dl, len(ds))]

    def test_no_duplicates_within_jpg_dataloaders(self, dl_cfg):
        (dl1, dl2, dl3), (ds1, ds2, ds3) = get_dataloaders(batch_size=16, **dl_cfg)
        for name, dl, ds in zip(
            ("train", "validation", "test"), (dl1, dl2, dl3), (ds1, ds2, ds3)
        ):
            hashes = self.collect_hashes_once(dl, ds)
            assert len(hashes) == len(ds)
            assert len(set(hashes)) == len(ds), (
                f"Duplicate hash found within '{name}' dataloader."
            )

    def test_no_duplicates_across_jpg_dataloaders(self, dl_cfg):
        (dl1, dl2, dl3), (ds1, ds2, ds3) = get_dataloaders(batch_size=16, **dl_cfg)
        names = ("train", "validation", "test")
        overall = {}
        for name, dl, ds in zip(names, (dl1, dl2, dl3), (ds1, ds2, ds3)):
            for h in self.collect_hashes_once(dl, ds):
                if h in overall:
                    raise AssertionError(
                        f"Duplicate found! Image in '{name}' was already in '{overall[h]}'."
                    )
                overall[h] = name
        total = len(ds1) + len(ds2) + len(ds3)
        assert len(overall) == total

    def test_aug_pipe_buffers_upsize(self, tmp_path):
        # Run data loader with tiny augmentation pipe. All buffers are too small.
        ds = get_npy_dataset(tmp_path)
        aug_pipe = m.DataAugmentationPipe([m.ResizeAugmentation(HEIGHT, WIDTH)], [16, 1, 1, 3], 1, 1)
        dl = DataLoader(ds, DEF_BATCH_SIZE, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, aug_pipe)
        batch, _ = dl.get_next_batch()
        assert 'np' in batch
        del batch

        # Then run an image raster dataloader. The buffers must upsize, otherwise the buffers overflow.
        _, dl = get_dataloader(DEF_BATCH_SIZE, DEF_NUM_THREADS, DEF_PREFETCH_SIZE)
        batch, _ = dl.get_next_batch()
        assert 'img' in batch


class TestDecoders:
    def test_exr(self):
        w, h = 500, 200
        img_data = np.random.rand(h, w, 3).astype(np.float32)
        cv2.imwrite(EXR_DATASET_FILE, img_data)

        ds = m.Dataset.from_subdirs("temp", [("img_exr", "img", m.ItemType.RASTER)], init_ds_fn)
        dl = DataLoader(ds, 1, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, get_pad_pipe())
        batch, _ = dl.get_next_batch()
        img = batch["img"][0, :h, :w]

        gt_img = cv2.imread(EXR_DATASET_FILE, cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        assert_low_error(img, gt_img)

    def test_png(self):
        w, h = 500, 200
        random_data = os.urandom(w * h * 3)
        img = Image.frombytes('RGB', (w, h), random_data)
        img.save(PNG_DATASET_FILE)

        ds = m.Dataset.from_subdirs("temp", [("img_png", "img", m.ItemType.RASTER)], init_ds_fn)
        dl = DataLoader(ds, 1, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, get_pad_pipe())
        batch, _ = dl.get_next_batch()
        img = batch["img"][0, :h, :w]

        pil_img = np.array(Image.open(PNG_DATASET_FILE).convert("RGB"), np.float32)
        pil_img = pil_img / 255.0

        assert_low_error(img, pil_img)

    def test_npy(self, tmp_path):
        ds = get_npy_dataset(tmp_path)
        dl = DataLoader(ds, 16, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, get_aug_pipe())

        batch, _ = dl.get_next_batch()
        np_data = batch["np"]
        assert np.all(np.ones((16, 3, 3, 4)) == np_data)

    def test_compressed(self, tmp_path):
        shape = (2, 3, 1)
        
        in_sub = tmp_path / "in"
        in_sub.mkdir()
        for i in range(10):
            arr = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + i)
            np.save(in_sub / f"file{i}.npy", arr)

        out_sub = tmp_path / "out"
        out_sub.mkdir()

        options = m.CompressorOptions(
            num_threads=4,
            input_directory=str(in_sub),
            output_directory=str(out_sub),
            shape=list(shape),
            cast_to_fp16=False,
            permutations=[[0, 1, 2]],
            with_bitshuffle=False,
            allowed_codecs=[],
            tolerance_for_worse_codec=0.01,
        )
        m.Compressor(options).start()

        mapping = [("out", "feat", m.ItemType.NONE)]
        ds = m.Dataset.from_subdirs(str(tmp_path), mapping, init_ds_fn)

        aug_pipe = m.DataAugmentationPipe(
            [m.PadAugmentation(shape[0], shape[1], m.PadSettings.PAD_BOTTOM_RIGHT)], 
            [16, shape[0], shape[1], shape[2]], 
            1, 4
        )

        dl = DataLoader(ds, 4, 4, 2, aug_pipe)

        batch, _ = dl.get_next_batch()
        feat = batch["feat"]
        
        assert feat.shape == (4, *shape)
        assert feat.dtype == np.float32

        for i in range(4):
            original_arr = np.load(in_sub / f"file{i}.npy")
            print(feat)
            print(original_arr)
            exit(0)
            assert np.allclose(feat[i], original_arr)


class TestPoints:
    def test_meta():
        # TODO: Points tests.
        raise ValueError()

# def test_end_to_end_perf(benchmark):
#     bs = 16
#     _, dl, _ = get_dataloader(batch_size=bs, num_threads=24, prefetch_size=3)

#     # Warmup a bit to fill prefetch and spin up threads
#     for _ in range(min(2, len(dl))):
#         _ = dl.get_next_batch()

#     num_batches = len(dl) * 16  # keep runtime reasonable

#     def fetch_loop():
#         items = 0
#         for _ in range(num_batches):
#             batch = dl.get_next_batch()
#             x = batch['img']
#             # Touch shape/size to ensure the object is realized and then drop it
#             items += int(x.size)
#             del batch
#         return items

#     total_items = benchmark(fetch_loop)
#     # Sanity check to keep benchmark from being optimized away
#     assert total_items > 0
