import argparse
import glob
import os
from typing import Dict

import numpy as np


def load_npy_files(directory: str, pattern: str, is_superpoint=False) -> Dict[str, np.ndarray]:
    files = glob.glob(os.path.join(directory, pattern))
    data = {}
    for file in files:
        base_name = os.path.basename(file).rsplit("_", 1)[0]
        loaded = np.load(file, allow_pickle=True)
        if is_superpoint:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                # not used here, but handle gracefully if .npz
                loaded = {k: loaded[k] for k in loaded.files}
            elif isinstance(loaded.item(), dict):
                loaded = loaded.item()
            data[base_name] = loaded["descriptors"]
        else:
            data[base_name] = np.array(loaded)
    return data


def build_dataset(dino_dir: str, superpoint_dir: str):
    dino_data = load_npy_files(dino_dir, "*_features.npy")
    superpoint_data = load_npy_files(superpoint_dir, "*_superpoint.npy", is_superpoint=True)
    # Intersect keys that exist in both
    keys = sorted(set(dino_data.keys()) & set(superpoint_data.keys()))
    X_list, y_list = [], []
    for k in keys:
        dino = dino_data[k]
        sp = superpoint_data[k]
        # Flatten dino to 1-D
        dino_vec = dino.reshape(-1)
        # Prepare 100x256 SP descriptors (pad/truncate)
        if sp.ndim == 1:
            sp = sp.reshape(1, -1)
        if sp.shape[0] > 100:
            sp = sp[:100]
        elif sp.shape[0] < 100:
            pad = np.zeros((100 - sp.shape[0], sp.shape[1]), dtype=sp.dtype)
            sp = np.concatenate([sp, pad], axis=0)
        sp_mean = sp.mean(axis=0)
        x = np.concatenate([dino_vec, sp_mean], axis=0)
        # Simple deterministic target: L2 norm of dino_vec
        y = np.linalg.norm(dino_vec).astype(np.float32)
        X_list.append(x)
        y_list.append(y)
    if not X_list:
        raise SystemExit("No overlapping feature files found to build dataset.")
    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list)[:, None]
    return X, y


def train_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1e-3):
    # Closed-form ridge regression: w = (X^T X + l2 I)^{-1} X^T y
    XTX = X.T @ X
    reg = l2 * np.eye(XTX.shape[0], dtype=XTX.dtype)
    w = np.linalg.solve(XTX + reg, X.T @ y)
    preds = X @ w
    mse = float(np.mean((preds - y) ** 2))
    return w, mse


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simple NumPy training over extracted features")
    ap.add_argument("--dino-dir", default="data/dino-features")
    ap.add_argument("--superpoint-dir", default="data/superpoint-features")
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args(argv)

    X, y = build_dataset(args.dino_dir, args.superpoint_dir)
    w, mse = train_ridge(X, y, l2=args.l2)
    print({"samples": int(X.shape[0]), "features": int(X.shape[1]), "mse": mse})


if __name__ == "__main__":
    main()
