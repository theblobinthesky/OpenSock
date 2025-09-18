import argparse
import glob
import os
from typing import Dict, Iterator, List, Any
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from model import ModelConfig, base_model_fuser, init_base_model_fuser


def create_dataloader(
    config: ModelConfig,
    camera_0_dino: str,
    camera_0_sp: str,
    camera_1_dino: str,
    camera_1_sp: str
) -> Iterator[Dict[str, Any]]:
    c0_dino_files = glob.glob(os.path.join(camera_0_dino, "*.npy"))
    c0_sp_files = glob.glob(os.path.join(camera_0_sp, "*.npy"))
    c1_dino_files = glob.glob(os.path.join(camera_1_dino, "*.npy"))
    c1_sp_files = glob.glob(os.path.join(camera_1_sp, "*.npy"))

    def get_basename(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    c0_dino_names = {get_basename(f): f for f in c0_dino_files}
    c0_sp_names = {get_basename(f): f for f in c0_sp_files}
    c1_dino_names = {get_basename(f): f for f in c1_dino_files}
    c1_sp_names = {get_basename(f): f for f in c1_sp_files}

    print(f"[debug]: {len(c0_dino_names)=}, {len(c0_sp_names)=}, {len(c1_dino_names)=}, {len(c1_sp_names)=}")

    common_names = set(c0_dino_names.keys()) & set(c0_sp_names.keys()) & set(c1_dino_names.keys()) & set(c1_sp_names.keys())
    if not common_names:
        raise ValueError("No matching quadruples found")

    sample_names = list(common_names)

    def collate(curr: List[Dict[str, Any]]) -> Dict[str, Any]:
        K = int(config.max_num_keyp)

        def pad_to_k(arr: jnp.ndarray) -> jnp.ndarray:
            n = arr.shape[0]
            d = arr.shape[1] if arr.ndim > 1 else 1
            take = min(n, K)
            arr_trunc = arr[:take]
            out = jnp.zeros((K, d), dtype=arr.dtype)
            out = out.at[:take].set(arr_trunc if d > 1 else arr_trunc.reshape(take, 1))
            return out

        c0_dino_list = []
        c0_points_list = []
        c0_features_list = []
        c0_counts_list = []
        c1_dino_list = []
        c1_points_list = []
        c1_features_list = []
        c1_counts_list = []

        for s in curr:
            # Pad arrays
            c0_dino_list.append(pad_to_k(s['c0_dino']))
            c0_points_list.append(pad_to_k(s['c0_points']))
            c0_features_list.append(pad_to_k(s['c0_features']))
            c1_dino_list.append(pad_to_k(s['c1_dino']))
            c1_points_list.append(pad_to_k(s['c1_points']))
            c1_features_list.append(pad_to_k(s['c1_features']))
            # Original counts (clipped to K for clarity in downstream use)
            c0_counts_list.append(min(s['c0_points'].shape[0], K))
            c1_counts_list.append(min(s['c1_points'].shape[0], K))

        return {
            'c0_dino': jnp.stack(c0_dino_list, axis=0),
            'c0_points': jnp.stack(c0_points_list, axis=0),
            'c0_features': jnp.stack(c0_features_list, axis=0),
            'c0_counts': jnp.asarray(c0_counts_list, dtype=jnp.int32),
            'c1_dino': jnp.stack(c1_dino_list, axis=0),
            'c1_points': jnp.stack(c1_points_list, axis=0),
            'c1_features': jnp.stack(c1_features_list, axis=0),
            'c1_counts': jnp.asarray(c1_counts_list, dtype=jnp.int32),
        }
   

    while True:
        np.random.shuffle(sample_names)
        batch_samples: List[Dict[str, Any]] = []

        for name in sample_names:
            try:
                c0_dino = np.load(c0_dino_names[name], allow_pickle=True)
                c0_sp = np.load(c0_sp_names[name], allow_pickle=True).item()
                c1_dino = np.load(c1_dino_names[name], allow_pickle=True)
                c1_sp = np.load(c1_sp_names[name], allow_pickle=True).item()

                c0_dino_flat = c0_dino.reshape(-1, c0_dino.shape[-1])
                c1_dino_flat = c1_dino.reshape(-1, c1_dino.shape[-1])

                batch_samples.append({
                    'c0_dino': jnp.array(c0_dino_flat),
                    'c0_points': jnp.array(c0_sp['points']),
                    'c0_features': jnp.array(c0_sp['features']),
                    'c1_dino': jnp.array(c1_dino_flat),
                    'c1_points': jnp.array(c1_sp['points']),
                    'c1_features': jnp.array(c1_sp['features'])
                })

                if len(batch_samples) >= config.batch_size:
                    curr = batch_samples[:config.batch_size]
                    batch = collate(curr)
                    # advance the ring buffer
                    batch_samples = batch_samples[config.batch_size:]
                    yield batch


            except Exception:
                continue

        if batch_samples:
            batch = collate(batch_samples)
            yield batch


@partial(jax.jit, static_argnames=['config'])
def train_step(config: ModelConfig, params: Dict, opt_state: optax.OptState, batch: Dict[str, Any]) -> tuple[Any, optax.OptState, Dict]:
    def loss_fn(params):
        local_feats_a = batch["c0_features"]
        points_a = batch["c0_points"]
        global_feats_a = batch["c0_dino"]
        counts_a = batch["c0_counts"]

        local_feats_b = batch["c1_features"]
        points_b = batch["c1_points"]
        global_feats_b = batch["c1_dino"]
        counts_b = batch["c1_counts"]

        sim_matrix = base_model_fuser(
            config, params,
            local_feats_a, points_a, global_feats_a, counts_a,
            local_feats_b, points_b, global_feats_b, counts_b
        )

        loss = jnp.array(1.0)
        return loss / len(batch)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optax.adam(1e-4).update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, {'loss': loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-0-dino", default="data/dataset/camera_0_dino")
    parser.add_argument("--camera-0-sp", default="data/dataset/camera_0_sp")
    parser.add_argument("--camera-1-dino", default="data/dataset/camera_1_dino")
    parser.add_argument("--camera-1-sp", default="data/dataset/camera_1_sp")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    config = ModelConfig()
    dataloader = create_dataloader(
        config,
        args.camera_0_dino, args.camera_0_sp,
        args.camera_1_dino, args.camera_1_sp
    )
    params = init_base_model_fuser(config)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(params)

    for epoch in range(args.epochs):
        for batch in dataloader:
            params, opt_state, metrics = train_step(config, params, opt_state, batch)
            print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
