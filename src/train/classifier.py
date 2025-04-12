import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax, jax.nn as nn, jax.numpy as jnp
import jax.nn.initializers as init
from .config import TrainConfig, DataConfig, ModelConfig
from .data import get_classifier_dataset
from .train import train_model, load_params
from .utils import focal_loss, accuracy, precision, recall


class Initializer:
    def __init__(self):
        self.key = jax.random.PRNGKey(0)
        self.kernel_init = init.glorot_uniform()
    
    def __call__(self, shape: tuple[int, ...]):
        k1, k2 = jax.random.split(self.key)
        self.key = k1
        return self.kernel_init(k2, shape)

init_fn = Initializer()


def linear_layer(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    x = x @ params['weights'] + params['biases']
    return x


def init_linear_layer(inp: int, out: int) -> jnp.ndarray:
    return {
        'weights': init_fn((inp, out)),
        'biases': jnp.zeros((out,))
    }


def linear_classifier(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    for i in range(6):
        x = linear_layer(params[f'linear{i}'], x)
        x = nn.gelu(x)

    x = linear_layer(params['out'], x)
    x = nn.sigmoid(x)

    return x


def init_linear_classifier():
    # Dinov2 dimensions by size: 384 (s), 768 (b), 1024 (l), 1536 (g)
    return {
        'linear0': init_linear_layer(768, 1024),
        'linear1': init_linear_layer(1024, 1024),
        'linear2': init_linear_layer(1024, 1024),
        'linear3': init_linear_layer(1024, 1024),
        'linear4': init_linear_layer(1024, 512),
        'linear5': init_linear_layer(512, 256),
        'out': init_linear_layer(256, 1)
    }


def train_classifier():
    dconfig = DataConfig()
    tconfig = TrainConfig()
    mconfig = ModelConfig()

    dataset = get_classifier_dataset(dconfig)
    params = init_linear_classifier()
    focal_loss_config = mconfig.classifier['focal_loss']

    def loss_fn(params, batch):
        target = batch['labels'] == 1.0
        pred = linear_classifier(params, batch['features'])
        
        return focal_loss(target, pred, focal_loss_config=focal_loss_config)

    def metric_fn(params, batch):
        target = batch['labels'] == 1.0
        pred = linear_classifier(params, batch['features'])

        return {
            'loss': focal_loss(target, pred, focal_loss_config=focal_loss_config),
            'accuracy': accuracy(target, pred),
            'precision': precision(target, pred),
            'recall': recall(target, pred)
        }

    train_model("classifier", tconfig, dconfig, mconfig, dataset, params, loss_fn, metric_fn)


if __name__ == "__main__":
    train_classifier()