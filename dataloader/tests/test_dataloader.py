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
    (8, 2),
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
        root_dir, [(sub_dir, "img", m.ItemType.RASTER)], init_ds_fn
    )


def get_npy_dataset(tmp_path, dtype=np.float32):
    subdir = tmp_path / "npy_subdir"
    subdir.mkdir()

    for i in range(16):
        testFile = tmp_path / "npy_subdir" / f"file{i}"
        np.save(testFile, np.ones((3, 3, 4), dtype=dtype))

    return m.Dataset.from_subdirs(
        str(tmp_path), [("npy_subdir", "np", m.ItemType.NONE)], init_ds_fn
    )


def get_aug_pipe():
    return m.DataAugmentationPipe(
        [m.ResizeAugmentation(HEIGHT, WIDTH)], [16, 500, 700, 3], 128, 4
    )


def get_pad_pipe():
    return m.DataAugmentationPipe(
        [m.PadAugmentation(1024, 1024, m.PadSettings.PAD_BOTTOM_RIGHT)],
        [16, 1024, 1024, 3],
        128,
        4,
    )


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
        (ds1, ds2, ds3),
    )


def assert_low_error(img, gt_img):
    err = np.mean(np.abs(img - gt_img))
    # if not np.all(err < 10 / 255.0):
    #     import matplotlib.pyplot as plt
    #     _, axs = plt.subplots(2, 1)
    #     axs[0].imshow(img)
    #     axs[1].imshow(gt_img)
    #     plt.show()
    assert np.all(err < 10 / 255.0), f"Error {err * 255.0:.1f} too high for image"


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


class TestRaster:
    def test_one_dataloader_once(self, dl_cfg):
        ds, dl = get_dataloader(batch_size=16, **dl_cfg)
        verify_correctness(ds, dl, bs=16)

    def test_one_dataloader_trice(self, dl_cfg):
        ds, dl = get_dataloader(batch_size=16, **dl_cfg)
        verify_correctness(ds, dl, bs=16, reps=3)

    def test_three_dlers_without_next_batch(self, dl_cfg):
        (dl1, dl2, dl3), (ds1, ds2, ds3) = get_dataloaders(batch_size=16, **dl_cfg)
        verify_correctness(ds1, dl1, bs=16)
        verify_correctness(ds2, dl2, bs=16)
        verify_correctness(ds3, dl3, bs=16)

    def test_three_dlers_with_next_batch(self, dl_cfg):
        bs = 16
        (dl1, dl2, dl3), (ds1, ds2, ds3) = get_dataloaders(batch_size=bs, **dl_cfg)
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
        aug_pipe = m.DataAugmentationPipe(
            [m.ResizeAugmentation(HEIGHT, WIDTH)], [16, 1, 1, 3], 1, 1
        )
        dl = DataLoader(
            ds, DEF_BATCH_SIZE, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, aug_pipe
        )
        batch, _ = dl.get_next_batch()
        assert "np" in batch
        del batch

        # Then run an image raster dataloader. The buffers must upsize, otherwise the buffers overflow.
        _, dl = get_dataloader(DEF_BATCH_SIZE, DEF_NUM_THREADS, DEF_PREFETCH_SIZE)
        batch, _ = dl.get_next_batch()
        assert "img" in batch


class TestDecoders:
    def test_exr(self):
        w, h = 500, 200
        img_data = np.random.rand(h, w, 3).astype(np.float32)
        cv2.imwrite(EXR_DATASET_FILE, img_data)

        ds = m.Dataset.from_subdirs(
            "temp", [("img_exr", "img", m.ItemType.RASTER)], init_ds_fn
        )
        dl = DataLoader(ds, 1, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, get_pad_pipe())
        batch, _ = dl.get_next_batch()
        img = batch["img"][0, :h, :w]

        gt_img = cv2.imread(EXR_DATASET_FILE, cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        assert_low_error(img, gt_img)

    def test_png(self):
        w, h = 500, 200
        random_data = os.urandom(w * h * 3)
        img = Image.frombytes("RGB", (w, h), random_data)
        img.save(PNG_DATASET_FILE)

        ds = m.Dataset.from_subdirs(
            "temp", [("img_png", "img", m.ItemType.RASTER)], init_ds_fn
        )
        dl = DataLoader(ds, 1, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, get_pad_pipe())
        batch, _ = dl.get_next_batch()
        img = batch["img"][0, :h, :w]

        pil_img = np.array(Image.open(PNG_DATASET_FILE).convert("RGB"), np.float32)
        pil_img = pil_img / 255.0

        assert_low_error(img, pil_img)

    @pytest.mark.parametrize(
        "dtype",
        # [np.float16, np.float32, np.float64, np.int32, np.uint8],
        [np.float32],
        ids=lambda d: str(np.dtype(d)),
    )
    def test_npy(self, tmp_path, dtype):
        if dtype == np.float64:
            import jax

            if not jax.config.read("jax_enable_x64"):
                pytest.skip("JAX x64 is disabled")

        ds = get_npy_dataset(tmp_path, dtype=dtype)
        dl = DataLoader(ds, 16, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, get_aug_pipe())

        batch, _ = dl.get_next_batch()
        np_data = batch["np"]
        expected = np.ones((16, 3, 3, 4), dtype=dtype)
        if dtype == np.uint8:
            expected = expected.astype(np.float32) / np.float32(255.0)
            assert np.allclose(expected, np_data)
        else:
            assert np.all(expected == np_data)

    def test_compressed(self, tmp_path):
        shape = (3, 3, 1)

        in_sub = tmp_path / "in"
        in_sub.mkdir()
        for i in range(10):
            arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + i
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
            1,
            4,
        )

        dl = DataLoader(ds, 4, 4, 2, aug_pipe)

        batch, _ = dl.get_next_batch()
        feat = batch["feat"]

        assert feat.shape == (4, *shape)
        assert feat.dtype == np.float32

        for i, paths in enumerate(ds.entries[:4]):
            npy_path = paths[0][: -len(".compressed")].replace("/out/", "/in/")
            original_arr = np.load(npy_path)
            assert np.allclose(feat[i], original_arr)


class TestPoints:
    MAX_POINTS = 16
    POINT_DIM = 3
    IMG_SIZE = 32

    def _get_pipe(self):
        return m.DataAugmentationPipe(
            [
                m.PadAugmentation(
                    self.IMG_SIZE, self.IMG_SIZE, m.PadSettings.PAD_BOTTOM_RIGHT
                )
            ],
            [16, self.IMG_SIZE, self.IMG_SIZE, 3],
            self.MAX_POINTS,
            4,
        )

    def _make_ds(self, tmp_path, lengths, mapping):
        root = tmp_path / "points_root"
        img_dir = root / "img_points"
        pts_dir = root / "pts_points"
        img_dir.mkdir(parents=True, exist_ok=True)
        pts_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(0)
        for idx, length in enumerate(lengths):
            if length > self.MAX_POINTS:
                raise ValueError("Sample length exceeds MAX_POINTS")

            img = (rng.random((self.IMG_SIZE, self.IMG_SIZE, 3)) * 255).astype(np.uint8)
            Image.fromarray(img).save(img_dir / f"sample_{idx}.png")

            pts = np.zeros((length, self.POINT_DIM), dtype=np.float32)
            if length:
                pts[:, 0] = 1.0
                pts[:, 1] = np.arange(length, dtype=np.float32)
                pts[:, 2] = float(idx)
            np.save(pts_dir / f"sample_{idx}.npy", pts)

        return m.Dataset.from_subdirs(str(root), mapping, init_ds_fn)

    def _idxs_in_ds_order(self, ds):
        idxs = []
        for entry in ds.entries:
            anchor = entry[0]
            name = os.path.basename(anchor)
            idxs.append(int(os.path.splitext(name)[0].split("_")[-1]))
        return idxs

    def _lengths_in_ds_order(self, ds, lengths):
        return [lengths[i] for i in self._idxs_in_ds_order(ds)]

    def test_points_must_follow_raster_item(self, tmp_path):
        lengths = [1, 2]
        mapping = [
            ("pts_points", "pts", m.ItemType.POINTS),
            ("img_points", "img", m.ItemType.RASTER),
        ]
        with pytest.raises(RuntimeError):
            self._make_ds(tmp_path, lengths, mapping)

    def test_points_batch_returns_lengths_metadata(self, tmp_path):
        lengths = [3, 7, 5, 1]
        mapping = [
            ("img_points", "img", m.ItemType.RASTER),
            ("pts_points", "pts", m.ItemType.POINTS),
        ]
        ds = self._make_ds(tmp_path, lengths, mapping)
        dl = DataLoader(
            ds, len(lengths), DEF_NUM_THREADS, DEF_PREFETCH_SIZE, self._get_pipe()
        )

        batch, metadata = dl.get_next_batch()

        assert "pts" in batch
        assert "pts" in metadata
        assert batch["pts"] is not None
        assert metadata["pts"] is not None
        pts = np.asarray(batch["pts"])
        assert pts.shape == (len(lengths), self.MAX_POINTS, self.POINT_DIM)
        assert pts.dtype == np.float32
        del pts  # give back control to the dataloader

        pts_lengths = np.asarray(metadata["pts"])
        assert pts_lengths.shape == (len(lengths),)
        assert pts_lengths.dtype == np.int32
        expected = np.array(
            self._lengths_in_ds_order(ds, lengths)[: len(lengths)], dtype=np.int32
        )
        assert np.array_equal(pts_lengths, expected)
        del pts_lengths  # give back control to the dataloader

    def test_points_tensor_prefix_matches_lengths(self, tmp_path):
        lengths = [2, 1, 3, 7]
        mapping = [
            ("img_points", "img", m.ItemType.RASTER),
            ("pts_points", "pts", m.ItemType.POINTS),
        ]
        ds = self._make_ds(tmp_path, lengths, mapping)
        dl = DataLoader(
            ds, len(lengths), DEF_NUM_THREADS, DEF_PREFETCH_SIZE, self._get_pipe()
        )

        batch, metadata = dl.get_next_batch()
        assert "pts" in batch
        assert "pts" in metadata
        assert batch["pts"] is not None
        assert metadata["pts"] is not None

        pts = np.asarray(batch["pts"])
        pts_lengths = np.asarray(metadata["pts"])

        idxs = self._idxs_in_ds_order(ds)[: len(lengths)]
        for b, (idx, length) in enumerate(zip(idxs, pts_lengths)):
            length = int(length)
            if length == 0:
                continue

            expected = np.zeros((length, self.POINT_DIM), dtype=np.float32)
            expected[:, 0] = 1.0
            expected[:, 1] = np.arange(length, dtype=np.float32)
            expected[:, 2] = float(idx)
            assert np.all(pts[b, :length] == expected)

    def test_points_metadata_matches_dataset_iteration_order(self, tmp_path):
        lengths = [2, 4, 6, 8, 3]
        batch_size = 3
        mapping = [
            ("img_points", "img", m.ItemType.RASTER),
            ("pts_points", "pts", m.ItemType.POINTS),
        ]
        ds = self._make_ds(tmp_path, lengths, mapping)
        dl = DataLoader(
            ds, batch_size, DEF_NUM_THREADS, DEF_PREFETCH_SIZE, self._get_pipe()
        )

        expected = self._lengths_in_ds_order(ds, lengths)
        offset = 0
        for _ in range(2):
            batch, metadata = dl.get_next_batch()
            assert "pts" in batch
            assert "pts" in metadata
            assert metadata["pts"] is not None

            pts_lengths = np.asarray(metadata["pts"])
            slice_expected = [
                expected[(offset + i) % len(expected)] for i in range(batch_size)
            ]
            assert np.array_equal(pts_lengths, np.array(slice_expected, dtype=np.int32))
            offset = (offset + batch_size) % len(expected)


def test_end_to_end_perf(benchmark):
    bs = DEF_BATCH_SIZE
    num_threads = DEF_NUM_THREADS
    prefetch_size = DEF_PREFETCH_SIZE

    cap = {"n_batches": 0, "in_bytes": 0}

    def setup_bench():
        ds, dl = get_dataloader(
            batch_size=bs, num_threads=num_threads, prefetch_size=prefetch_size
        )

        # Warmup a bit to fill prefetch and spin up threads.
        for _ in range(min(2, len(dl))):
            _ = dl.get_next_batch()

        n_batches = min(len(dl), 32)
        paths = [item[0] for item in ds.entries]
        in_bytes = 0
        for i in range(n_batches * bs):
            in_bytes += os.path.getsize(paths[i % len(paths)])

        cap["n_batches"] = n_batches
        cap["in_bytes"] = in_bytes
        return (dl, n_batches), {}

    def run_bench(dl, n_batches):
        items = 0
        for _ in range(n_batches):
            batch, _ = dl.get_next_batch()
            x = batch["img"]

            # Ensure the returned tensor is realized.
            if hasattr(x, "block_until_ready"):
                x.block_until_ready()

            items += int(x.size)
        return items

    benchmark.group = "Dataloader"
    total_items = benchmark.pedantic(
        run_bench,
        setup=setup_bench,
        rounds=3,
        iterations=1,
    )

    out_bytes = cap["n_batches"] * bs * HEIGHT * WIDTH * 3 * 4
    benchmark.extra_info["batches"] = int(cap["n_batches"])
    benchmark.extra_info["items"] = int(total_items)
    mean = benchmark.stats["mean"] if benchmark.stats else None
    benchmark.extra_info["in_MBps"] = (cap["in_bytes"] / (1024 * 1024)) / mean
    benchmark.extra_info["out_MBps"] = (out_bytes / (1024 * 1024)) / mean
    assert total_items > 0
