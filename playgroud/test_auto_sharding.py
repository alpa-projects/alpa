from functools import partial
import os

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import optim
from flax.core.frozen_dict import FrozenDict, freeze
from paranum import parallelize


def test_matmul():

    @parallelize
    def matmul(a, b):
        a =  a @ b
        return a

    x = jnp.ones((128, 128))
    y = jnp.ones((128, 128))

    c = matmul(x, y)
    c.block_until_ready()

    np.testing.assert_allclose(np.array(c), np.array(x @ y))


def test_mlp():
    #os.environ['XLA_FLAGS'] = '--xla_disable_hlo_passes=auto_sharding'
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'

    class Model(nn.Module):
        hidden_dim: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
            return x


    @parallelize
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch['x'])
            return jnp.mean((out - batch['y']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer


    batch_size = 32
    hidden_dim = 512
    input_dim = output_dim = hidden_dim

    x = jnp.ones((batch_size, input_dim))
    y = jnp.ones((batch_size, output_dim))

    model = Model(hidden_dim=hidden_dim)
    params = FrozenDict({
        "params": {
            "Dense_0": {
                "kernel": jnp.ones((input_dim, hidden_dim)),
            },
            "Dense_1": {
                "kernel": jnp.ones((hidden_dim, output_dim)),
            }
        }
    })
    optimizer = optim.GradientDescent(1e-2).create(params)

    optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)


if __name__ == "__main__":
    #test_matmul()

    test_mlp()

