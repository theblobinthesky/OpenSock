import jax
import math
import jax.nn as nn
import jax.numpy as jnp
import jax.lax as lax
import jax.nn.initializers as init
from typing import Dict


class ModelConfig:
    def __init__(self):
        self.max_num_keyp = 1024
        self.keyp_reduced_fraction = 0.5

        self.batch_size = 16

        self.num_fourier_series_terms_for_pos_enc_per_dim = 7
        self.num_pos_enc_mlps = 4

        self.num_blocks = 16
        self.local_feature_dim = 256

    @property
    def max_num_reduced_keyp(self):
        return int(math.ceil(self.max_num_keyp * self.keyp_reduced_fraction))


class Initializer:
    def __init__(self):
        self.key = jax.random.PRNGKey(0)
        self.kernel_init = init.glorot_uniform()

    def __call__(self, shape: tuple[int, ...]):
        k1, k2 = jax.random.split(self.key)
        self.key = k1
        return self.kernel_init(k2, shape)


init_fn = Initializer()


def init_mlp(inp: int, out: int) -> dict:
    return {
        "weights": init_fn((inp, out)),
        "biases": jnp.zeros((out,))
    }


def mlp(
    params: dict,
    x: jnp.ndarray,  # (N, config.local_feature_dim)
    do_activation: bool = False
) -> jnp.ndarray:
    sh = x.shape[:-1]
    s = math.prod(sh)

    x = x.reshape((s, -1)) @ params["weights"] + params["biases"]

    if do_activation:
        x = jax.nn.gelu(x)

    return x.reshape((*sh, -1))

# Per token.


def layer_norm(x: jnp.ndarray, epsilon: float = 1e-6) -> jnp.ndarray:
    var = jnp.var(x, axis=-1, keepdims=True)
    x = x - jnp.mean(x, axis=-1, keepdims=True)
    return x / jnp.sqrt(var + epsilon)

def self_attention(
    config: ModelConfig,
    K: jnp.ndarray,
    V: jnp.ndarray,
    Q: jnp.ndarray,
    counts_KVQ: jnp.ndarray,
) -> jnp.ndarray:
    kvq_dim = K.shape[1]
    assert kvq_dim == Q.shape[1]
    bs = config.batch_size

    assert len(K.shape) == 3
    assert len(Q.shape) == 3

    Kt = K.transpose((0, 2, 1))
    att = (Q @ Kt) / math.sqrt(config.local_feature_dim)

    # Delete contrib. of disabled keys.
    indices = jnp.repeat(jnp.arange(kvq_dim)[None, None, :], bs, axis=0)
    att = jnp.where(indices < counts_KVQ[:, None, None], att, -1e30)

    att = nn.softmax(att, axis=-1)
    att = att @ V

    # Delete contrib. of disabled queries.
    # TODO: This is not strictly necessary.
    indices = jnp.repeat(jnp.arange(kvq_dim)[None, :, None], bs, axis=0)
    att = jnp.where(indices < counts_KVQ[:, None, None], att, 0.0)
    return att
 
def cross_attention(
    config: ModelConfig,
    K: jnp.ndarray,         # (M, C)
    V: jnp.ndarray,         # (M, C)
    Q: jnp.ndarray,         # (N, C)
    counts_KV: jnp.ndarray,   # (M,)
    counts_Q: jnp.ndarray     # (N,)
) -> jnp.ndarray:
    q_dim = Q.shape[1]
    bs = config.batch_size

    assert len(K.shape) == 4 and len(V.shape) == 4
    assert len(Q.shape) == 3

    Q = Q[:, :, None, :]
    Kt = K.transpose((0, 1, 3, 2))
    att = (Q @ Kt) / math.sqrt(config.local_feature_dim)
    att = att.squeeze(axis=2)

    # Delete contrib. of disabled keys.
    indices = jnp.arange(config.max_num_reduced_keyp)[None, None, :]
    indices = jnp.repeat(indices, bs, axis=0)
    indices = jnp.repeat(indices, config.max_num_keyp, axis=1)
    att = jnp.where(indices < counts_KV[:, None, None], att, -1e30)

    att = nn.softmax(att, axis=-1)
    att = att[:, :, None, :] @ V
    att = att.squeeze(axis=2)

    # Delete contrib. of disabled queries.
    # TODO: This is not strictly necessary.
    indices = jnp.repeat(jnp.arange(q_dim)[None, :, None], bs, axis=0)
    att = jnp.where(indices < counts_Q[:, None, None], att, 0.0)

    return att


def init_intra_graph_attention(config: ModelConfig) -> dict:
    return {
        "key_ll": init_mlp(config.local_feature_dim, config.local_feature_dim),
        "value_ll": init_mlp(config.local_feature_dim, config.local_feature_dim),
        "query_ll": init_mlp(config.local_feature_dim, config.local_feature_dim),
        "mlp": init_mlp(config.local_feature_dim, config.local_feature_dim)
    }


def intra_graph_attention(
    config: ModelConfig, params: dict,
    local_feats: jnp.ndarray,   # (N, C)
    pos_feats: jnp.ndarray,     # (N, C)
    counts: jnp.ndarray
) -> jnp.ndarray:

    local_feats = layer_norm(local_feats)  # pre-norm formulation
    K = mlp(params["key_ll"], local_feats + pos_feats)
    V = mlp(params["value_ll"], local_feats)
    Q = mlp(params["query_ll"], local_feats + pos_feats)

    local_feats_change = self_attention(config, K, V, Q, counts)
    local_feats = local_feats + local_feats_change

    local_feats = layer_norm(local_feats)  # pre-norm formulation
    return local_feats + mlp(params["mlp"], local_feats, do_activation=True)


def init_inter_graph_attention(config: ModelConfig) -> dict:
    return {
        "mlp-key": init_mlp(config.local_feature_dim, config.local_feature_dim),
        "mlp-value": init_mlp(config.local_feature_dim, config.local_feature_dim),
        "mlp-query": init_mlp(config.local_feature_dim, config.local_feature_dim),
        "mlp-residual": init_mlp(config.local_feature_dim * 2, config.local_feature_dim),
    }


def inter_graph_attention(
    config: ModelConfig, params: dict,

    local_feats_a: jnp.ndarray,
    pos_feats_a: jnp.ndarray,
    counts_a: jnp.ndarray,

    local_feats_b: jnp.ndarray,
    pos_feats_b: jnp.ndarray,
    red_counts_b: jnp.ndarray,

    indices_a_b: jnp.ndarray
) -> jnp.ndarray:
    local_feats_a = layer_norm(local_feats_a)
    local_feats_b = layer_norm(local_feats_b)

    local_feats_b_reord = jnp.take_along_axis(
        local_feats_b[:, None, :, :],
        indices_a_b[:, :, :, None],
        axis=2
    )

    pos_feats_b_reord = jnp.take_along_axis(
        pos_feats_b[:, None, :, :],
        indices_a_b[:, :, :, None],
        axis=2
    )

    K = mlp(params["mlp-key"], local_feats_b_reord + pos_feats_b_reord)
    V = mlp(params["mlp-value"], local_feats_b_reord)
    Q = mlp(params["mlp-query"], local_feats_a + pos_feats_a)
    local_feats_a_change = cross_attention(config, K, V, Q, red_counts_b, counts_a)

    cc = jnp.concatenate([local_feats_a, local_feats_a_change], axis=-1)
    cc = layer_norm(cc)
    return local_feats_a + mlp(params["mlp-residual"], cc, do_activation=True)


def init_info_propagation_block(config: ModelConfig) -> dict:
    return {
        "intra": init_intra_graph_attention(config),
        "inter": init_inter_graph_attention(config)
    }


def info_propagation_block(
    config: ModelConfig, params: dict,
    local_feats_a: jnp.ndarray,   # (A, C)
    pos_feats_a: jnp.ndarray,     # (A, C)
    counts_a: jnp.ndarray,
    local_feats_b: jnp.ndarray,   # (B, C)
    pos_feats_b: jnp.ndarray,     # (B, C)
    counts_b: jnp.ndarray,
    indices_a_b: jnp.ndarray,
    indices_b_a: jnp.ndarray,
    red_counts_a: jnp.ndarray,
    red_counts_b: jnp.ndarray
) -> jnp.ndarray:

    local_feats_a = intra_graph_attention(
        config, params["intra"], local_feats_a, pos_feats_a, counts_a)

    local_feats_b = intra_graph_attention(
        config, params["intra"], local_feats_b, pos_feats_b, counts_b)

    local_feats_a = inter_graph_attention(
        config, params["inter"],
        local_feats_a, pos_feats_a, counts_a,
        local_feats_b, pos_feats_b, red_counts_b,
        indices_a_b
    )

    local_feats_b = inter_graph_attention(
        config, params["inter"],
        local_feats_b, pos_feats_b, counts_b,
        local_feats_a, pos_feats_a, red_counts_a,
        indices_b_a
    )

    return local_feats_a, local_feats_b


def build_inter_image_graphs(
    config: ModelConfig,
    global_feats_a: jnp.ndarray,  # (A, D)
    counts_a: jnp.ndarray,        # (A,)
    global_feats_b: jnp.ndarray,  # (B, D)
    counts_b: jnp.ndarray         # (B,)
) -> jnp.ndarray:

    def build_a_to_b_graph(global_feats_1, global_feats_2, counts_2):
        def norm_lengths(feats: jnp.ndarray) -> jnp.ndarray:
            return feats / jnp.linalg.norm(feats, axis=-1, keepdims=True)
        norm_global_1 = norm_lengths(global_feats_1)
        norm_global_2 = norm_lengths(global_feats_2)
        sims = norm_global_1 @ norm_global_2.transpose((0, 2, 1))

        # Disable contribution of non-points.
        indices = jnp.repeat(jnp.arange(config.max_num_keyp)[
                             None, None, :], config.batch_size, axis=0)
        counts_2 = counts_2[:, None, None]
        sims = jnp.where(indices < counts_2, sims, -2.0)

        # Reorder features.
        N = config.max_num_reduced_keyp
        indices = jnp.argsort(-sims, axis=-1)[..., :N]
        return indices

    indices_a_b = build_a_to_b_graph(global_feats_a, global_feats_b, counts_b)
    indices_b_a = build_a_to_b_graph(global_feats_b, global_feats_a, counts_a)
    counts_a = jnp.ceil(
        counts_a * config.keyp_reduced_fraction).astype("int16")
    counts_b = jnp.ceil(
        counts_b * config.keyp_reduced_fraction).astype("int16")

    return indices_a_b, indices_b_a, counts_a, counts_b


def pos_enc_1d(config: ModelConfig, vals: jnp.ndarray) -> jnp.ndarray:
    batch = vals.shape[0]
    vals = vals.flatten()

    L = config.num_fourier_series_terms_for_pos_enc_per_dim
    freqs = jnp.pi * 2.0 ** jnp.arange(L)
    angles = vals[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=0).reshape((batch, -1, 2 * L))


def pos_enc_2d_separable(config: ModelConfig, points: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([pos_enc_1d(config, points[..., 0]), pos_enc_1d(config, points[..., 1])], axis=-1)


def init_pos_enc_2d_learned(config: ModelConfig) -> dict:
    assert config.num_pos_enc_mlps >= 1
    return {
        f"mlp-init": init_mlp(config.num_fourier_series_terms_for_pos_enc_per_dim * 4, config.local_feature_dim)
    } | {
        f"mlp-{i}": init_mlp(config.local_feature_dim, config.local_feature_dim)
        for i in range(config.num_pos_enc_mlps - 1)
    }


def pos_enc_2d_learned(config: ModelConfig, params: dict, points: jnp.ndarray) -> jnp.ndarray:
    enc = pos_enc_2d_separable(config, points)

    enc = mlp(params["mlp-init"], enc, do_activation=True)
    for i in range(config.num_pos_enc_mlps - 1):
        enc = mlp(params[f"mlp-{i}"], enc, do_activation=True)

    return enc


def init_base_model_fuser(config: ModelConfig) -> dict:
    return {
        "pos-enc": init_pos_enc_2d_learned(config)
    } | {
        f"block-{i}": init_info_propagation_block(config)
        for i in range(config.num_blocks)
    }


def base_model_fuser(
    config: ModelConfig, params: dict,

    local_feats_a: jnp.ndarray,   # (A, C)
    points_a: jnp.ndarray,        # (A, 2)
    global_feats_a: jnp.ndarray,  # (A, D)
    counts_a: jnp.ndarray,

    local_feats_b: jnp.ndarray,   # (B, C)
    points_b: jnp.ndarray,        # (B, 2)
    global_feats_b: jnp.ndarray,  # (B, D)
    counts_b: jnp.ndarray
) -> jnp.ndarray:

    indices_a_b, indices_b_a, red_counts_a, red_counts_b = build_inter_image_graphs(
        config,
        global_feats_a, counts_a,
        global_feats_b, counts_b
    )

    pos_feats_a = pos_enc_2d_learned(config, params["pos-enc"], points_a)
    pos_feats_b = pos_enc_2d_learned(config, params["pos-enc"], points_b)

    # Incrementally refine local descriptors.
    for i in range(config.num_blocks):
        local_feats_a, local_feats_b = info_propagation_block(
            config, params[f"block-{i}"],
            local_feats_a, pos_feats_a, counts_a,
            local_feats_b, pos_feats_b, counts_b,
            indices_a_b, indices_b_a,
            red_counts_a, red_counts_b
        )

    # Build similarity matrix.
    sim = local_feats_a @ local_feats_b.transpose((0, 2, 1))
    return sim
