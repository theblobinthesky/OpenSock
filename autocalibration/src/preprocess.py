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


def run(camera_0_dir: str, camera_1_dir: str, camera_0_npy: str, camera_1_npy: str):
    # Prepare outputs
    shutil.rmtree(camera_0_npy, ignore_errors=True)
    shutil.rmtree(camera_1_npy, ignore_errors=True)
    ensure_dir(camera_0_npy)
    ensure_dir(camera_1_npy)

    # Process camera_0
    process_directory(camera_0_dir, camera_0_npy, "camera_0")

    # Process camera_1
    process_directory(camera_1_dir, camera_1_npy, "camera_1")


def process_directory(input_dir: str, output_dir: str, label: str):
    # Collect image files
    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, e)))
    image_paths.sort()

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for path in image_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")

        dino_feat = compute_dino_like_feature(img)
        np.save(os.path.join(output_dir, f"{base}_features.npy"), dino_feat)

        sp_desc = compute_superpoint_like_descriptors(img)
        sp_payload = {
            "descriptors": sp_desc,
            # Provide placeholders for compatibility with prior structure
            "keypoints": np.zeros((sp_desc.shape[0], 2), dtype=np.float32),
            "scores": np.ones((sp_desc.shape[0],), dtype=np.float32),
        }
        np.save(os.path.join(output_dir, f"{base}_superpoint.npy"), sp_payload)
        print(f"Processed {label}: {base}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lightweight feature extraction (offline)")
    ap.add_argument("--camera-0", default="data/dataset/camera_0")
    ap.add_argument("--camera-1", default="data/dataset/camera_1")
    ap.add_argument("--camera-0-npy", default="data/dataset/camera_0_npy")
    ap.add_argument("--camera-1-npy", default="data/dataset/camera_1_npy")
    args = ap.parse_args(argv)
    run(args.camera_0, args.camera_1, args.camera_0_npy, args.camera_1_npy)


if __name__ == "__main__":
    main()
