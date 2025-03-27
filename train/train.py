import jax, optax, pickle, os
from tqdm import tqdm
import jax.numpy as jnp
from config import TrainConfig, DataConfig, ModelConfig
# from data import Dataset
from logger import Logger, MockLogger
from typing import Callable
from functools import partial
from collections import Counter

@partial(jax.jit, static_argnames=['optim', 'loss_fn'])
def _train_step(params, batch, opt_state, optim, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, opt_state

def _save_model(params: dict, path: str):
    with open(path, "wb") as file:
        pickle.dump(params, file)

def load_params(path: str):
    with open(path, "rb") as file:
        return pickle.load(file)

def train_model(purpose: str,
                tconfig: TrainConfig,
                dconfig: DataConfig,
                mconfig: ModelConfig,
                train_dataset_tf,
                valid_dataset_tf,
                params: dict,
                loss_fn: Callable[[dict[str, jnp.ndarray], dict[str, jnp.ndarray]], jnp.ndarray],
                validation_fn: Callable[[dict[str, jnp.ndarray], dict[str, jnp.ndarray]], jnp.ndarray]
                ):
    optim = optax.adam(learning_rate=tconfig.lr)
    opt_state = optim.init(params)
    last_valid_loss = float('inf')

    logger = Logger(purpose, dconfig, tconfig, mconfig) if 'LOG' in os.environ else MockLogger()

    for epoch in range(tconfig.num_epochs):
        # train_dataset = iter(train_dataset_tf)
        # valid_dataset = iter(valid_dataset_tf)

        epoch_loss = 0.0
        tqdm_iter = tqdm(range(100))#len(train_dataset_tf)))
        for _ in tqdm_iter:
            # image, (bboxes, classes) = next(train_dataset)
            # batch = {"img": image, "bboxes": bboxes, "classes": classes}
            # batch = {k: jnp.array(v.numpy()) for k, v in batch.items()}
            # print({k: v.shape for k, v in batch.items()})
            batch = {
                "img": jnp.zeros((1, 64, 64, 3)),
                "bboxes": jnp.zeros((1, 30, 4)),
                "classes": jnp.zeros((1, 30))
            }

            loss, params, opt_state = _train_step(params, batch, opt_state, optim, loss_fn)

            epoch_loss += loss
            tqdm_iter.set_description(f"Training loss: {loss:.4f}")
        epoch_loss /= len(train_dataset_tf)

        # valid_metrics = None
        # tqdm_iter = tqdm(range(len(valid_dataset_tf)))
        # for _ in tqdm(range(len(valid_dataset_tf))):
        #     image, (bboxes, classes) = next(valid_dataset)
        #     batch = {"img": image, "bboxes": bboxes, "classes": classes}
        #     batch = {k: jnp.array(v.numpy()) for k, v in batch.items()}

        #     metrics = validation_fn(params, batch)
        #     if valid_metrics is None: valid_metrics = Counter(metrics)
        #     else: valid_metrics += metrics

        #     tqdm_iter.set_description(f"Validation loss: {metrics['loss']:.4f}")

        valid_metrics = {key: value / len(valid_dataset_tf) for key, value in valid_metrics.items()}
        valid_loss = valid_metrics['loss']

        if (improv := last_valid_loss > valid_loss):
            _save_model(params, tconfig.model_path)
            last_valid_loss = valid_loss

        logger.log({"train_loss": epoch_loss, "validation": valid_metrics})
        tqdm.write(f"Epoch {epoch + 1}, Training loss: {epoch_loss:.4f}, Validation loss: {valid_loss:.4f}" + (", Saved model" if improv else ""))