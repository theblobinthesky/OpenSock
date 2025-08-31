import argparse
import glob
import os
import shutil

import numpy as np
from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_dino_like_feature(img: Image.Image) -> np.ndarray:
    # Lightweight global descriptor: grayscale -> 16x16 -> flatten (256)
    g = img.convert("L").resize((16, 16), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    vec = arr.flatten()[None, :]  # shape (1, 256)
    return vec


def compute_superpoint_like_descriptors(img: Image.Image, patches=100, patch_size=16) -> np.ndarray:
    # Deterministic grid sampling patches on a 64x64 grayscale image
    g = img.convert("L").resize((64, 64), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    # choose a 10x10 grid of top-lefts over [0..64-patch_size]
    grid_n = int(np.ceil(np.sqrt(patches)))
    coords = np.linspace(0, 64 - patch_size, grid_n).astype(int)
    descs = []
    for yi in coords:
        for xi in coords:
            patch = arr[yi : yi + patch_size, xi : xi + patch_size]
            d = patch.flatten()
            # normalize per descriptor
            if d.std() > 1e-6:
                d = (d - d.mean()) / (d.std() + 1e-6)
            descs.append(d.astype(np.float32))
            if len(descs) >= patches:
                break
        if len(descs) >= patches:
            break
    descs = np.stack(descs, axis=0)  # (M, 256)
    return descs


def run(inputs: str, dino_out: str, sp_out: str):
    # Prepare outputs
    shutil.rmtree(dino_out, ignore_errors=True)
    shutil.rmtree(sp_out, ignore_errors=True)
    ensure_dir(dino_out)
    ensure_dir(sp_out)

    # Collect image files
    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(inputs, e)))
    image_paths.sort()

    if not image_paths:
        print(f"No images found in {inputs}")
        return

    for path in image_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")

        dino_feat = compute_dino_like_feature(img)
        np.save(os.path.join(dino_out, f"{base}_features.npy"), dino_feat)

        sp_desc = compute_superpoint_like_descriptors(img)
        sp_payload = {
            "descriptors": sp_desc,
            # Provide placeholders for compatibility with prior structure
            "keypoints": np.zeros((sp_desc.shape[0], 2), dtype=np.float32),
            "scores": np.ones((sp_desc.shape[0],), dtype=np.float32),
        }
        np.save(os.path.join(sp_out, f"{base}_superpoint.npy"), sp_payload)
        print(f"Processed {base}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lightweight feature extraction (offline)")
    ap.add_argument("--inputs", default="data/inputs")
    ap.add_argument("--dino-out", default="data/dino-features")
    ap.add_argument("--sp-out", default="data/superpoint-features")
    args = ap.parse_args(argv)
    run(args.inputs, args.dino_out, args.sp_out)


if __name__ == "__main__":
    main()
