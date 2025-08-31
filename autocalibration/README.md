Autocalibration (Simplified Pipeline)

Overview

- Offline-friendly end-to-end pipeline using lightweight feature extraction and NumPy-based training.
- Avoids external tools and heavyweight ML frameworks (Blender, Torch, TF, JAX, HF models).

Quick Start

- Inputs: place images under `data/inputs/` (PNG/JPG).
- Run pipeline: `python -m src.pipeline`
- Optional flags:
  - `--skip-preprocess`: reuse existing features in `data/dino-features` and `data/superpoint-features`.
  - `--skip-train`: only extract features.
  - `--inputs`, `--dino-out`, `--sp-out`: configure directories.

What It Does

- Preprocess (`src/preprocess.py`):
  - Computes a compact global descriptor per image (grayscale 16x16 flattened) saved as `*_features.npy`.
  - Computes 100 simple patch descriptors (16x16 grayscale patches flattened) saved as `*_superpoint.npy` with keys `descriptors`, `keypoints`, `scores`.

- Train (`src/train.py`):
  - Builds a dataset by concatenating the global descriptor with the mean of the 100 patch descriptors.
  - Trains a small ridge regression model in closed-form and reports MSE.

Notes

- Existing heavy scripts (`src/scraper.py`, `src/blender_render.py`) are left intact but are not part of the default pipeline flow.
- Dependencies are kept minimal: NumPy, Pillow, tqdm, requests.
