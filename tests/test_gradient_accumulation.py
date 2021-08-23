"""Test auto sharding with simple computational graphs."""

import os
import unittest

import numpy as np

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp

from parax import (parallelize, set_parallelize_options, mark_gradient,
                   testing, global_config)


def all_reduce_cost(num_devices, num_bytes):
    return 2.0 * (num_devices - 1) / num_devices * num_bytes


class AutoShardingBasicTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        set_parallelize_options(jax.devices()[:4])

    def test_gradient_accumulation(self):
        batch_size = 16
        num_micro_batches = 2
        hidden_size = 64
        use_bias = False

        class Model(nn.Module):
            @nn.compact
            def __call__(self, x, deterministic):
                x = nn.Dense(hidden_size, use_bias=use_bias)(x)
                #x = nn.Dense(hidden_size, use_bias=use_bias)(x)
                return x

        x = jnp.ones((batch_size, hidden_size))
        y = jnp.ones((batch_size, hidden_size))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, True)
        optimizer = optim.Momentum(1e-2).create(params)
        #optimizer = optim.Adam(1e-2).create(params)

        @parallelize
        def func(optimizer, batch):
            def loss_func(params):
                out = model.apply(params, batch['x'], False)
                return jnp.mean((out - batch['y'])**2)

            grad = jax.grad(loss_func)(optimizer.target)
            grad = mark_gradient(grad)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        global_config.num_micro_batches = num_micro_batches
        func(optimizer, {"x": x, "y": y})


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingBasicTest("test_gradient_accumulation"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
