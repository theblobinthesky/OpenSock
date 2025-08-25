#!/usr/bin/env python3
import os
import io
import tarfile
from typing import List, Tuple

import numpy as np
from PIL import Image

import requests
import native_dataloader as m


# Config
HEIGHT, WIDTH = 300, 300
NUM_THREADS, PREFETCH_SIZE = 16, 16

JPG_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
JPG_DATASET_DIR = os.path.join("temp", "jpg_dataset")


def ensure_jpg_dataset() -> Tuple[str, str]:
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
    return (
        m.Dataset.from_subdirs(
            root_dir,
            [m.Head(m.FileType.JPG, "img", (HEIGHT, WIDTH, 3))],
            [sub_dir],
            init_ds_fn,
        ),
        root_dir,
    )


def get_dataloader(batch_size: int):
    ds, root_dir = get_dataset()
    dl = m.DataLoader(ds, batch_size, NUM_THREADS, PREFETCH_SIZE)
    print(f"{len(dl)=}, {batch_size=}, {len(dl) % batch_size=}")
    return ds, dl, root_dir


def _collect_for_reps(
    ds, dl, root_dir: str, bs: int, reps: int, threshold: float = 10.0 / 255.0
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Collect dataloader and ground-truth batches with continuous wrap across reps."""
    # Flat list of absolute filepaths for the 'img' head
    entries = [f"{root_dir}{item[0]}" for item in ds.entries]

    dataloader_batches: List[np.ndarray] = []
    gt_batches: List[np.ndarray] = []
    row_labels: List[str] = []

    per_rep_batches = len(dl)
    total_batches = per_rep_batches * reps
    for batch_idx in range(total_batches):
        batch = dl.get_next_batch()
        imgs = batch["img"]

        # Build ground-truth images for this batch using continuous wrapping
        gt_imgs = []
        N = len(entries)
        start = (batch_idx * bs) % N
        for i in range(bs):
            path = entries[(start + i) % N]
            pil_img = Image.open(path).convert("RGB")
            pil_img = np.array(pil_img.resize((WIDTH, HEIGHT)), np.float32) / 255.0
            gt_imgs.append(pil_img)

        gt_batch = (
            np.stack(gt_imgs, axis=0) if gt_imgs else np.zeros((0, HEIGHT, WIDTH, 3), dtype=np.float32)
        )

        dataloader_batches.append(imgs)
        gt_batches.append(gt_batch)
        rep_idx = batch_idx // per_rep_batches
        batch_in_rep = batch_idx % per_rep_batches
        row_labels.append(f"({rep_idx}, {batch_in_rep})")

    return dataloader_batches, gt_batches, row_labels


def _plot_batch_grid(batches: List[np.ndarray], row_labels: List[str], title: str, out_path: str) -> None:
    import matplotlib.pyplot as plt

    nrows = len(batches)
    ncols = max((b.shape[0] for b in batches), default=1)

    fig_w = max(8, min(16, ncols))
    fig_h = max(8, min(24, nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
    if ncols == 1:
        axs = np.expand_dims(axs, axis=1)

    for r, batch in enumerate(batches):
        for c in range(ncols):
            ax = axs[r][c]
            if c < batch.shape[0]:
                ax.imshow(np.clip(batch[c], 0.0, 1.0))
                if c == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                else:
                    ax.axis("off")
            else:
                ax.set_visible(False)
        left_ax = axs[r][0]
        left_ax.set_ylabel(row_labels[r], rotation=0, ha="right", va="center", labelpad=14, fontsize=9)

    fig.suptitle(title)
    plt.tight_layout()
    fig.subplots_adjust(left=0.12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(bs: int = 16, reps: int = 2) -> None:
    ds_viz, dl_viz, root_dir_viz = get_dataloader(batch_size=bs)
    dl_batches, gt_batches, row_labels = _collect_for_reps(
        ds_viz, dl_viz, root_dir_viz, bs=bs, reps=reps
    )

    out_dir = os.path.join("temp", "ui")
    dl_out = os.path.join(out_dir, "compare_reps2_dataloader.png")
    gt_out = os.path.join(out_dir, "compare_reps2_groundtruth.png")

    _plot_batch_grid(dl_batches, row_labels, f"Dataloader Output (reps={reps})", dl_out)
    _plot_batch_grid(gt_batches, row_labels, f"Ground Truth (reps={reps})", gt_out)

    print(f"Saved: {dl_out}")
    print(f"Saved: {gt_out}")


if __name__ == "__main__":
    # Simple arg parse for optional overrides
    import argparse

    parser = argparse.ArgumentParser(description="Compare dataloader vs GT for reps=2 until mismatch")
    parser.add_argument("--batch_size", "--bs", type=int, default=16)
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()

    main(bs=args.batch_size, reps=args.reps)
