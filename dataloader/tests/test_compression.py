import os
import io
import tarfile
import requests
import numpy as np
from PIL import Image
import native_dataloader as m
import pytest
import time
from dataclasses import dataclass

JPG_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
JPG_DATASET_DIR = os.path.join("temp", "jpg_dataset")
FEATURES_DIR = os.path.join("temp", "features")

HEIGHT, WIDTH = 300, 300
D_OUT = 32
NUM_THREADS = 4

DINO_REPO = "facebookresearch/dinov2"
DINO_MODEL = "dinov2_vitb14_reg"
PATCH_SIZE = 14
EMBED_DIM = 768


def _ensure_dirs():
    os.makedirs("temp", exist_ok=True)
    os.makedirs(JPG_DATASET_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)


def ensure_jpg_dataset():
    _ensure_dirs()
    if not os.listdir(JPG_DATASET_DIR):
        resp = requests.get(JPG_DATASET_URL)
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            tar.extractall(path=JPG_DATASET_DIR, filter=lambda ti, _: ti)
    return os.path.join(JPG_DATASET_DIR, "flower_photos")


def list_some_jpgs(root, subfolder="daisy", limit=2):
    folder = os.path.join(root, subfolder)
    imgs = []
    for name in os.listdir(folder):
        if name.lower().endswith((".jpg", ".jpeg")):
            imgs.append(os.path.join(folder, name))
            if len(imgs) >= limit:
                break
    return imgs


def imread_rgb_jpg(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _load_dino_or_skip():
    torch = pytest.importorskip("torch", reason="PyTorch required.")
    # try:
    model = torch.hub.load(DINO_REPO, DINO_MODEL)
    # except Exception as e:
    #     pytest.skip(f"Could not load {DINO_REPO}:{DINO_MODEL} ({e}).")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    return model, dev


def _dense_features_dino(img_rgb, model, dev):
    import torch
    H, W, _ = img_rgb.shape
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    t = (t - mean) / std
    pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = t.shape[-2], t.shape[-1]
    with torch.no_grad():
        out = model.forward_features(t)
        tok = out.get("x_norm_patchtokens", None) if isinstance(out, dict) else out
        if tok is None and isinstance(out, dict):
            tok = out.get("x_prenorm", None)
        if tok is None:
            pytest.skip("Model did not expose patch tokens.")
        gh = Hp // PATCH_SIZE
        gw = Wp // PATCH_SIZE
        if tok.shape[1] == gh * gw + 1:
            tok = tok[:, 1:, :]
        tok = tok.reshape(1, gh, gw, EMBED_DIM).permute(0, 3, 1, 2)
        torch.manual_seed(0)
        P = torch.randn(EMBED_DIM, D_OUT, device=dev)
        P = P / (P.norm(dim=0, keepdim=True) + 1e-8)
        feat = torch.einsum("bdhw,dk->bkhw", tok, P)
        up = torch.nn.functional.interpolate(feat, size=(Hp, Wp), mode="bilinear", align_corners=False)
        up = up[:, :, :H, :W]
        dense = up[0].permute(1, 2, 0).contiguous().float().cpu().numpy()
        return dense


def compute_and_save_features_jpgs(image_paths, features_dir=FEATURES_DIR):
    os.makedirs(features_dir, exist_ok=True)
    model, dev = _load_dino_or_skip()
    saved = []
    for p in image_paths:
        img = imread_rgb_jpg(p)
        arr = _dense_features_dino(img, model, dev)
        assert arr.shape == (HEIGHT, WIDTH, D_OUT) and arr.dtype == np.float32
        out_name = os.path.splitext(os.path.basename(p))[0] + ".npy"
        out_path = os.path.join(features_dir, out_name)
        np.save(out_path, arr)
        saved.append(out_path)
    return saved


def ensure_features_prepared(limit: int=64):
    if not os.path.isdir(FEATURES_DIR) or not any(f.endswith(".npy") for f in os.listdir(FEATURES_DIR)):
        root = ensure_jpg_dataset()
        jpgs = list_some_jpgs(root, subfolder="daisy", limit=limit)
        assert jpgs, "No JPGs found."
        compute_and_save_features_jpgs(jpgs, FEATURES_DIR)


def test_prepare_dense_features_jpg_only():
    ensure_features_prepared()
    files = [os.path.join(FEATURES_DIR, f) for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")]
    assert files
    x = np.load(files[0])
    assert x.shape == (HEIGHT, WIDTH, D_OUT)
    assert x.dtype == np.float32


def assert_allclose_after_compress_decompress(outputDir: str, shape: list[int], options: m.CompressorOptions, original_dtype: str="float16"):
    compressor = m.Compressor(options)
    compressor.start()
    outs = [f for f in os.listdir(FEATURES_DIR)]
    assert outs

    decompressor = m.Decompressor(shape)
    for file_name in outs:
        compressed_path = f"{outputDir}/{file_name}.compressed"
        npy_path = f"{FEATURES_DIR}/{file_name}"

        decompressed_arr = decompressor.decompress(compressed_path)
        original_arr = np.astype(np.load(npy_path), original_dtype)
        assert np.allclose(decompressed_arr, original_arr)


def test_compress_fp16(tmp_path):
    ensure_features_prepared()

    shape = [HEIGHT, WIDTH, D_OUT]
    options = m.CompressorOptions(
        num_threads=NUM_THREADS,
        input_directory=FEATURES_DIR,
        output_directory=str(tmp_path),
        shape=shape,
        cast_to_fp16=True,
        permutations=[[0, 1, 2]],
        with_bitshuffle=False,
        allowed_codecs=[],
        tolerance_for_worse_codec=0.01,
    )
    assert_allclose_after_compress_decompress(str(tmp_path), shape, options)


def test_compress_fp32(tmp_path):
    ensure_features_prepared()

    shape = [HEIGHT, WIDTH, D_OUT]
    options = m.CompressorOptions(
        num_threads=NUM_THREADS,
        input_directory=FEATURES_DIR,
        output_directory=str(tmp_path),
        shape=shape,
        cast_to_fp16=False,
        permutations=[[0, 1, 2]],
        with_bitshuffle=False,
        allowed_codecs=[],
        tolerance_for_worse_codec=0.01,
    )
    assert_allclose_after_compress_decompress(str(tmp_path), shape, options, original_dtype="float32")


def test_compress_fp16_permute(tmp_path):
    ensure_features_prepared()

    shape = [HEIGHT, WIDTH, D_OUT]
    options = m.CompressorOptions(
        num_threads=NUM_THREADS,
        input_directory=FEATURES_DIR,
        output_directory=str(tmp_path),
        shape=shape,
        cast_to_fp16=True,
        permutations=[[0, 1, 2], [2, 0, 1]],
        with_bitshuffle=False,
        allowed_codecs=[],
        tolerance_for_worse_codec=0.01,
    )
    assert_allclose_after_compress_decompress(str(tmp_path), shape, options)


def test_compress_fp32_permute(tmp_path):
    ensure_features_prepared()

    shape = [HEIGHT, WIDTH, D_OUT]
    options = m.CompressorOptions(
        num_threads=NUM_THREADS,
        input_directory=FEATURES_DIR,
        output_directory=str(tmp_path),
        shape=shape,
        cast_to_fp16=False,
        permutations=[[0, 1, 2], [2, 0, 1]],
        with_bitshuffle=False,
        allowed_codecs=[],
        tolerance_for_worse_codec=0.01,
    )
    assert_allclose_after_compress_decompress(str(tmp_path), shape, options, original_dtype="float32")


def test_compress_fp16_permute_bitshuffle(tmp_path):
    ensure_features_prepared()

    shape = [HEIGHT, WIDTH, D_OUT]
    options = m.CompressorOptions(
        num_threads=NUM_THREADS,
        input_directory=FEATURES_DIR,
        output_directory=str(tmp_path),
        shape=[HEIGHT, WIDTH, D_OUT],
        cast_to_fp16=True,
        permutations=[[0, 1, 2], [2, 0, 1]],
        with_bitshuffle=True,
        allowed_codecs=[],
        tolerance_for_worse_codec=0.01,
    )
    assert_allclose_after_compress_decompress(str(tmp_path), shape, options)


def test_compress_fp16_permute_bitshuffle_compress(tmp_path):
    ensure_features_prepared()

    shape = [HEIGHT, WIDTH, D_OUT]
    options = m.CompressorOptions(
        num_threads=NUM_THREADS,
        input_directory=FEATURES_DIR,
        output_directory=str(tmp_path),
        shape=[HEIGHT, WIDTH, D_OUT],
        cast_to_fp16=True,
        permutations=[[0, 1, 2], [2, 0, 1]],
        with_bitshuffle=True,
        allowed_codecs=[m.Codec.ZSTD_LEVEL_3, m.Codec.ZSTD_LEVEL_7],
        tolerance_for_worse_codec=0.01,
    )
    assert_allclose_after_compress_decompress(str(tmp_path), shape, options)


def _gather_raw_feature_files():
    return [os.path.join(FEATURES_DIR, f)
            for f in os.listdir(FEATURES_DIR)
            if f.endswith(".npy")]

def _humansize(n: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024 or unit == "TB":
            return f"{n:6.2f} {unit}"
        n /= 1024.0
    return f"{n:6.2f} TB"

@dataclass(frozen=True)
class BenchConfig:
    name: str
    fp16: bool
    bitshuffle: bool
    codecs: list  # list[m.Codec]
    permutations: list[list[int]]

@dataclass
class BenchResult:
    cfg: BenchConfig
    compressed_bytes: int
    n_out: int
    comp_secs: float
    decomp_secs: float

def _run_compress(shape: list[int], out_dir: str, cfg: BenchConfig) -> BenchResult:
    os.makedirs(out_dir, exist_ok=True)
    options = m.CompressorOptions(
        num_threads=NUM_THREADS,
        input_directory=FEATURES_DIR,
        output_directory=out_dir,
        shape=shape,
        cast_to_fp16=cfg.fp16,
        permutations=cfg.permutations,
        with_bitshuffle=cfg.bitshuffle,
        allowed_codecs=cfg.codecs,
        tolerance_for_worse_codec=0.01,
    )
    t0 = time.perf_counter()
    m.Compressor(options).start()
    t1 = time.perf_counter()

    outs = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".compressed")]
    assert outs, f"No compressed outputs found in {out_dir}"
    comp_bytes = sum(os.path.getsize(p) for p in outs)

    # Decompression timing (single pass, one Decompressor instance)
    decomp = m.Decompressor(shape)
    t2 = time.perf_counter()
    for p in outs:
        _ = decomp.decompress(p)
    t3 = time.perf_counter()

    return BenchResult(cfg=cfg,
                       compressed_bytes=comp_bytes,
                       n_out=len(outs),
                       comp_secs=(t1 - t0),
                       decomp_secs=(t3 - t2))

@pytest.mark.order(-1)
def test_benchmark_compression_and_decode(tmp_path, capsys=None):
    ensure_features_prepared()

    raw_files = _gather_raw_feature_files()
    assert raw_files, "No raw feature files to benchmark."
    raw_total = sum(os.path.getsize(p) for p in raw_files)

    shape = [HEIGHT, WIDTH, D_OUT]
    perms_all = [
        [0, 1, 2], [0, 2, 1],
        [1, 0, 2], [1, 2, 0],
        [2, 0, 1], [2, 1, 0],
    ]

    configs = [
        BenchConfig("fp32_zstd7",              fp16=False, bitshuffle=False, codecs=[m.Codec.ZSTD_LEVEL_7],              permutations=perms_all),
        BenchConfig("fp16_zstd7",              fp16=True,  bitshuffle=False, codecs=[m.Codec.ZSTD_LEVEL_7],              permutations=perms_all),
        BenchConfig("fp32_bshuf_zstd7",        fp16=False, bitshuffle=True,  codecs=[m.Codec.ZSTD_LEVEL_7],              permutations=perms_all),
        BenchConfig("fp16_bshuf_zstd7",        fp16=True,  bitshuffle=True,  codecs=[m.Codec.ZSTD_LEVEL_7],              permutations=perms_all),
        BenchConfig("fp16_bshuf_zstd(3,7,22)", fp16=True,  bitshuffle=True,  codecs=[m.Codec.ZSTD_LEVEL_3, m.Codec.ZSTD_LEVEL_7, m.Codec.ZSTD_LEVEL_22], permutations=perms_all),
        BenchConfig("fp16_bshuf_zstd22",       fp16=True,  bitshuffle=True,  codecs=[m.Codec.ZSTD_LEVEL_22],             permutations=perms_all),
    ]

    results: list[BenchResult] = []
    for cfg in configs:
        out_dir = tmp_path / f"bench_{cfg.name}"
        res = _run_compress(shape, str(out_dir), cfg)
        assert res.n_out == len(raw_files), f"Mismatch: {res.n_out=} vs {len(raw_files)=}"
        results.append(res)

    # Derive metrics
    n_elems = HEIGHT * WIDTH * D_OUT
    table_rows = []
    for r in results:
        ratio = r.compressed_bytes / max(1, raw_total)
        # Throughputs
        comp_MBps_in  = (raw_total / (1024*1024)) / r.comp_secs
        comp_MBps_out = (r.compressed_bytes / (1024*1024)) / r.comp_secs
        out_dtype_bytes = 2 if r.cfg.fp16 else 4
        decomp_in_MBps  = (r.compressed_bytes / (1024*1024)) / r.decomp_secs
        decomp_out_MBps = ((r.n_out * n_elems * out_dtype_bytes) / (1024*1024)) / r.decomp_secs

        table_rows.append({
            "name": r.cfg.name,
            "fp16": r.cfg.fp16,
            "bshuf": r.cfg.bitshuffle,
            "codecs": ",".join([c.name for c in r.cfg.codecs]) if r.cfg.codecs else "RAW",
            "compressed": r.compressed_bytes,
            "ratio": ratio,
            "comp_s": r.comp_secs,
            "c_in_MBps": comp_MBps_in,
            "c_out_MBps": comp_MBps_out,
            "dec_s": r.decomp_secs,
            "d_in_MBps": decomp_in_MBps,
            "d_out_MBps": decomp_out_MBps,
        })

    # Pretty print
    header = (
        f"{'config':26} {'fp16':>5} {'bshuf':>5} {'codecs':22} "
        f"{'compressed':>12} {'ratio':>7}  "
        f"{'comp(s)':>7} {'c_in MB/s':>10} {'c_out MB/s':>11}  "
        f"{'dec(s)':>7} {'d_in MB/s':>10} {'d_out MB/s':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in table_rows:
        print(
            f"{row['name']:26} {str(row['fp16']):>5} {str(row['bshuf']):>5} {row['codecs']:22} "
            f"{_humansize(row['compressed']):>12} {row['ratio']:7.3f}  "
            f"{row['comp_s']:7.3f} {row['c_in_MBps']:10.2f} {row['c_out_MBps']:11.2f}  "
            f"{row['dec_s']:7.3f} {row['d_in_MBps']:10.2f} {row['d_out_MBps']:11.2f}"
        )
    print(f"{'raw total:':26} {'':>5} {'':>5} {'':22} {_humansize(raw_total):>12} {1.000:7.3f}")

    assert any(row["ratio"] < 1.0 for row in table_rows), "No configuration achieved compression < 1.0x of raw .npy"

    if capsys is not None:
        captured = capsys.readouterr()
        print(captured.out)
