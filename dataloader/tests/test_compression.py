import os
import io
import tarfile
import requests
import numpy as np
from PIL import Image
import pytest
import native_dataloader as m

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


def ensure_features_prepared():
    if not os.path.isdir(FEATURES_DIR) or not any(f.endswith(".npy") for f in os.listdir(FEATURES_DIR)):
        root = ensure_jpg_dataset()
        jpgs = list_some_jpgs(root, subfolder="daisy", limit=2)
        assert jpgs, "No JPGs found."
        compute_and_save_features_jpgs(jpgs, FEATURES_DIR)


def test_prepare_dense_features_jpg_only():
    ensure_features_prepared()
    files = [os.path.join(FEATURES_DIR, f) for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")]
    assert files
    x = np.load(files[0])
    assert x.shape == (HEIGHT, WIDTH, D_OUT)
    assert x.dtype == np.float32


def assert_allclose_after_compress_decompress(shape: list[int], options: m.CompressorOptions, original_dtype: str="float16"):
    compressor = m.Compressor(options)
    compressor.start()
    outs = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".compressed")]
    assert outs

    decompressor = m.Decompressor(shape)
    for file_name in outs:
        compressed_path = f"{FEATURES_DIR}/{file_name}"
        npy_path = compressed_path.replace(".compressed", "")

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
    assert_allclose_after_compress_decompress(shape, options)


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
    assert_allclose_after_compress_decompress(shape, options, original_dtype="float32")


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
    assert_allclose_after_compress_decompress(shape, options)


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
    assert_allclose_after_compress_decompress(shape, options, original_dtype="float32")


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
    assert_allclose_after_compress_decompress(shape, options)


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
    assert_allclose_after_compress_decompress(shape, options)

