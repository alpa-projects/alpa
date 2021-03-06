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

from paranum import parallelize
from utils import DataLoader, abstract_eval


class Model(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        #x = nn.Dense(features=self.hidden_size)(x)
        return x


@parallelize
def create_train_state(model, batch):
    rngkey = jax.random.PRNGKey(0)
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

    return optimizer

def main():
    batch_size = 1024
    hidden_size = 1 << 10

    n_epoch = 2
    n_batch = 2

    train_loader = DataLoader(batch_size, hidden_size, n_batch)

    print("Init model")
    model = Model(hidden_size=hidden_size)

    train_state = create_train_state(
        model,
        {"x": jnp.ones((batch_size, hidden_size)),
         "y": jnp.ones((batch_size, hidden_size))}
    )

    print("Train")
    for epoch in range(n_epoch):
        for batch, (x, y) in enumerate(train_loader):
            tic = time.time()
            train_state = train_step(train_state, {"x": x, "y": y}, model.apply)

            print("Epoch: %d\tBatch: %d\tTime: %.2f" % (epoch, batch, time.time() - tic))


if __name__ == "__main__":
    main()

