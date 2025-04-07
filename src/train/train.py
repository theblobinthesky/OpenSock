import threading, sys, select
import jax, optax, pickle, os
from tqdm import tqdm
import jax.numpy as jnp
from native_dataloader import Dataset, DataLoader
from typing import Callable
from functools import partial
from collections import Counter
from .config import TrainConfig, DataConfig, ModelConfig
from .logger import Logger, MockLogger
from .utils import infinite_learning_rate_scheduler

stop_training = False

def monitor_input():
    global stop_training
    print("Type 's' then Enter at any time to stop training.")
    while not stop_training:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() == 's':
                stop_training = True
                print("Stop signal received. Training will halt after current epoch.")
                break

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

def test_one_epoch(
        purpose: str,
        params: dict,
        data_loader: DataLoader,
        metric_fn: Callable[[dict[str, jnp.ndarray], dict[str, jnp.ndarray]], jnp.ndarray]
        ):
    acc_metrics = None
    tqdm_iter = tqdm(range(len(data_loader)))
    for _ in tqdm_iter:
        batch = data_loader.get_next_batch()

        metrics = metric_fn(params, batch)
        if acc_metrics is None: acc_metrics = Counter(metrics)
        else: acc_metrics += metrics

        tqdm_iter.set_description(f"{purpose} loss: {metrics['loss']:.4f}")

    acc_metrics = {key: value / len(data_loader) for key, value in acc_metrics.items()}
    return acc_metrics

def train_model(purpose: str,
                tconfig: TrainConfig,
                dconfig: DataConfig,
                mconfig: ModelConfig,
                dataset: Dataset,
                params: dict,
                loss_fn: Callable[[dict[str, jnp.ndarray], dict[str, jnp.ndarray]], jnp.ndarray],
                metric_fn: Callable[[dict[str, jnp.ndarray], dict[str, jnp.ndarray]], jnp.ndarray]
                ):
    threading.Thread(target=monitor_input, daemon=True).start()

    optim = optax.inject_hyperparams(optax.adam)(learning_rate=0.0)
    opt_state = optim.init(params)
    last_valid_loss = float('inf')

    train_ds, valid_ds, test_ds = dataset.split_train_validation_test(dconfig.split_train_percentage, dconfig.split_valid_percentage)
    train_dl = DataLoader(train_ds, tconfig.batch_size, tconfig.train_dl_num_threads, tconfig.train_dl_prefetch_size)
    valid_dl = DataLoader(valid_ds, tconfig.batch_size, tconfig.valid_dl_num_threads, tconfig.valid_dl_prefetch_size)
    logger = Logger(purpose, dconfig, tconfig, mconfig) if 'LOG' in os.environ else MockLogger()

    lr_schedule = {
        'max_lr': tconfig.lr_schedule['max_lr'],
        'const_lr': tconfig.lr_schedule['const_lr'],
        'min_lr':  tconfig.lr_schedule['min_lr'],
        'num_warmup_steps':  int(tconfig.lr_schedule['num_warmup_epochs'] * len(train_dl)),
        'num_cooldown_steps': int(tconfig.lr_schedule['num_cooldown_epochs'] * len(train_dl)),
        'num_decay_steps':  int(tconfig.lr_schedule['num_decay_epochs'] * len(train_dl)),
        'num_annealing_steps': int(tconfig.lr_schedule['num_annealing_epochs'] * len(train_dl))
    }

    step = 0
    for epoch in range(tconfig.num_epochs):
        epoch_loss = 0.0
        tqdm_iter = tqdm(range(len(train_dl)), desc=f"Training epoch {epoch + 1}/{tconfig.num_epochs}")
        for _ in tqdm_iter:
            lr = infinite_learning_rate_scheduler(lr_schedule, step)
            logger.log({"lr": lr})
            opt_state.hyperparams['learning_rate'] = lr
            step += 1

            batch = train_dl.get_next_batch()
            loss, params, opt_state = _train_step(params, batch, opt_state, optim, loss_fn)
            epoch_loss += loss

            tqdm_iter.set_description(f"Training loss: {loss:.4f}")
        epoch_loss /= len(train_dl)

        valid_metrics = test_one_epoch("Validation", params, valid_dl, metric_fn)
        valid_loss = valid_metrics['loss']

        if (improv := last_valid_loss > valid_loss):
            _save_model(params, tconfig.model_path)
            last_valid_loss = valid_loss

        logger.log({"train_loss": epoch_loss, "validation": valid_metrics})
        tqdm.write(f"Epoch {epoch + 1}, Training loss: {epoch_loss:.4f}, Validation loss: {valid_loss:.4f}" + (", Saved model" if improv else ""))

        if stop_training:
            print("Stopping training early...")
            break


    # Test:
    test_dl = DataLoader(test_ds, tconfig.batch_size, tconfig.valid_dl_num_threads, tconfig.valid_dl_prefetch_size)
    test_metrics = test_one_epoch("Testing", params, test_dl, metric_fn)
    logger.log({"test": test_metrics})
