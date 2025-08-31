import os
import glob
import numpy as np
import tensorflow as tf
import wandb
import jax
import jax.numpy as jnp
import optax
from dotenv import load_dotenv
load_dotenv()

def load_npy_files(directory, pattern, is_superpoint=False):
    files = glob.glob(os.path.join(directory, pattern))
    data = {}
    for file in files:
        base_name = os.path.basename(file).rsplit('_', 1)[0]
        loaded = np.load(file, allow_pickle=True)
        if is_superpoint:
            data[base_name] = loaded.item()['descriptors']
        else:
            data[base_name] = loaded
    return data

def create_dataset(dino_dir, superpoint_dir):
    dino_data = load_npy_files(dino_dir, "*_features.npy")
    superpoint_data = load_npy_files(superpoint_dir, "*_superpoint.npy", is_superpoint=True)
    
    keys = list(dino_data.keys())
    
    def generator():
        for key in keys:
            dino = dino_data[key]
            superpoint = superpoint_data[key]
            yield dino, superpoint
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(1, 256, 768), dtype=tf.float32), tf.TensorSpec(shape=(None, 256), dtype=tf.float32)))
    dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)
    return dataset

def init_params(rng):
    rng1, rng2 = jax.random.split(rng)
    w1 = jax.random.normal(rng1, (256*768 + 100*256, 128)) * 0.1
    b1 = jnp.zeros(128)
    w2 = jax.random.normal(rng2, (128, 1)) * 0.1
    b2 = jnp.zeros(1)
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

def model(params, dino, superpoint):
    dino_flat = dino.reshape((dino.shape[0], -1))
    superpoint_flat = superpoint.reshape((superpoint.shape[0], -1))
    concat = jnp.concatenate([dino_flat, superpoint_flat], axis=-1)
    hidden = jax.nn.relu(concat @ params['w1'] + params['b1'])
    output = hidden @ params['w2'] + params['b2']
    return output

def create_train_state(rng, learning_rate):
    params = init_params(rng)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    return params, opt_state, optimizer

# @jax.jit
def train_step(params, opt_state, optimizer, dino, superpoint, target):
    def loss_fn(p):
        preds = model(p, dino, superpoint)
        return jnp.mean((preds - target) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def main():
    wandb.init(project="autocalibration", config={"learning_rate": 0.001, "epochs": 10})
    
    dino_dir = "data/dino-features"
    superpoint_dir = "data/superpoint-features"
    
    dataset = create_dataset(dino_dir, superpoint_dir)
    
    rng = jax.random.PRNGKey(0)
    params, opt_state, optimizer = create_train_state(rng, wandb.config.learning_rate)
    
    for epoch in range(wandb.config.epochs):
        epoch_loss = 0
        num_batches = 0
        for batch in dataset:
            dino_batch, superpoint_batch = batch
            dino_batch = jnp.array(dino_batch.numpy())
            superpoint_batch = jnp.array(superpoint_batch.numpy())
            if superpoint_batch.shape[1] > 100:
                superpoint_batch = superpoint_batch[:, :100]
            elif superpoint_batch.shape[1] < 100:
                pad = jnp.zeros((superpoint_batch.shape[0], 100 - superpoint_batch.shape[1], 256))
                superpoint_batch = jnp.concatenate([superpoint_batch, pad], axis=1)

            # Dummy target
            target = jax.random.uniform(rng, (dino_batch.shape[0], 1))
            params, opt_state, loss = train_step(params, opt_state, optimizer, dino_batch, superpoint_batch, target)
            epoch_loss += loss
            num_batches += 1
            rng, _ = jax.random.split(rng)
        avg_loss = epoch_loss / num_batches
        wandb.log({"loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch}, Loss: {avg_loss}")

if __name__ == "__main__":
    main()
