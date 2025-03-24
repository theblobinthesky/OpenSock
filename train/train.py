import jax, optax, pickle, os
from tqdm import tqdm
import jax.numpy as jnp
from config import TrainConfig, DataConfig
from data import Dataset
from logger import Logger, MockLogger
from typing import Callable
from functools import partial

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

def train_model(tconfig: TrainConfig,
                dconfig: DataConfig,
                train_dataset: Dataset,
                valid_dataset: Dataset,
                params: dict,
                loss_fn: Callable[[dict[str, jnp.ndarray], dict[str, jnp.ndarray]], jnp.ndarray],
                ):
    optim = optax.adam(learning_rate=tconfig.lr)
    opt_state = optim.init(params)
    last_valid_loss = float('inf')

    logger = Logger(dconfig, tconfig) if 'LOG' in os.environ else MockLogger()

    for epoch in range(tconfig.num_epochs):
        epoch_loss = 0.0
        iter = tqdm(range(len(train_dataset)))
        for _ in iter:
            batch = train_dataset.get_next_batch()
            batch = {k: jnp.array(v) for k, v in batch.items()}

            loss, params, opt_state = _train_step(params, batch, opt_state, optim, loss_fn)

            epoch_loss += loss
            iter.set_description(f"Training loss: {loss:.4f}")
        epoch_loss /= len(train_dataset)

        valid_loss = 0.0
        for _ in range(len(valid_dataset)):
            batch = valid_dataset.get_next_batch()
            batch = {k: jnp.array(v) for k, v in batch.items()}

            loss = jax.jit(loss_fn)(params, batch)
            valid_loss += loss
        valid_loss /= len(valid_dataset)

        if (improv := last_valid_loss > valid_loss):
            _save_model(params, tconfig.model_path)
            last_valid_loss = valid_loss

        logger.log({"train_loss": epoch_loss, "valid_loss": valid_loss})
        tqdm.write(f"Epoch {epoch + 1}, Training loss: {epoch_loss:.4f}, Validation loss: {valid_loss:.4f}" + (", Saved model" if improv else ""))