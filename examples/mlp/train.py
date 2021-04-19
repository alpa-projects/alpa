from functools import partial
import time
from typing import Any, Callable, Sequence, Optional

import numpy as np
import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim

from parax import parallelize, annotate_gradient, compute_bytes
from utils import DataLoader

GB = 1 << 30


class Model(nn.Module):
    hidden_dim: int
    kernel_init: str = "zeros"

    @nn.compact
    def __call__(self, x):
        if self.kernel_init == "zeros":
            kernel_init = flax.linen.initializers.zeros
        else:
            kernel_init = flax.linen.linear.default_kernel_init

        x = nn.Dense(features=self.hidden_dim,
                     kernel_init=kernel_init, use_bias=False)(x)
        #x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim,
                     kernel_init=kernel_init, use_bias=False)(x)
        return x


#@parallelize
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
    grad = annotate_gradient(grad)
    new_optimizer = optimizer.apply_gradient(grad)

    return new_optimizer


def block_until_ready(train_state):
    train_state.target['params']['Dense_0']['kernel'].block_until_ready()


def main():
    batch_size = 128
    hidden_dim = 1024
    input_dim = output_dim = hidden_dim

    n_epoch = 1
    n_batch = 10

    train_loader = DataLoader(batch_size, input_dim, output_dim, n_batch)

    print("Init model")
    model = Model(hidden_dim=hidden_dim)

    rngkey = jax.random.PRNGKey(0)
    train_state = create_train_state(
        rngkey,
        model,
        {"x": jnp.ones((batch_size, input_dim)),
         "y": jnp.ones((batch_size, output_dim))}
    )

    block_until_ready(train_state)
    print(f"Train state size: {compute_bytes(train_state) / GB: .2f} GB")

    print("Train")
    for epoch in range(n_epoch):
        costs = []
        for batch, (x, y) in enumerate(train_loader):
            x.block_until_ready()

            tic = time.time()
            train_state = train_step(train_state, {"x": x, "y": y}, model.apply)
            block_until_ready(train_state)
            toc = time.time()

            costs.append(toc - tic)

        costs = np.array(costs[2:]) * 1e3
        print(costs)
        print("Mean cost: %.2f ms (std: %.2f ms)" % (np.median(costs), np.std(costs)))


if __name__ == "__main__":
    main()

