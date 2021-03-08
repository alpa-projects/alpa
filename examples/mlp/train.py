import functools
import time
from typing import Any, Callable, Sequence, Optional

import numpy as np
import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim

from paranum import parallelize, data_parallel, compute_bytes
from utils import DataLoader

GB = 1 << 30

class Model(nn.Module):
    hidden_size: int
    kernel_init: str = "zeros"

    @nn.compact
    def __call__(self, x):
        if self.kernel_init == "zeros":
            kernel_init = flax.linen.initializers.zeros
        else:
            kernel_init = flax.linen.linear.default_kernel_init

        x = nn.Dense(features=self.hidden_size,
                     kernel_init=kernel_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size,
                     kernel_init=kernel_init)(x)
        return x


@parallelize
def create_train_state(rngkey, model, batch):
    params = model.init(rngkey, batch['x'])
    optimizer = optim.GradientDescent(1e-2).create(params)
    return optimizer


@parallelize
def train_step(optimizer, batch, apply_fn):
    def loss_func(params):
        out = apply_fn(params, batch['x'])
        return jnp.mean((out - batch['y']) ** 2)

    grad = jax.grad(loss_func)(optimizer.target)
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=0.1)

    return new_optimizer


def main():
    batch_size = 1024
    hidden_size = (1 << 10)

    n_epoch = 2
    n_batch = 2

    train_loader = DataLoader(batch_size, hidden_size, n_batch)

    print("Init model")
    model = Model(hidden_size=hidden_size)

    rngkey = jax.random.PRNGKey(0)
    train_state = create_train_state(
        rngkey,
        model,
        {"x": jnp.ones((batch_size, hidden_size)),
         "y": jnp.ones((batch_size, hidden_size))}
    )

    train_state.target['params']['Dense_0']['kernel'].block_until_ready()
    print(f"Total size: {compute_bytes(train_state) / GB: .2f} GB")
    exit()

    print("Train")
    for epoch in range(n_epoch):
        for batch, (x, y) in enumerate(train_loader):
            tic = time.time()
            train_state = train_step(train_state, {"x": x, "y": y}, model.apply)
            print("Epoch: %d\tBatch: %d\tTime: %.2f" % (epoch, batch, time.time() - tic))


if __name__ == "__main__":
    main()

