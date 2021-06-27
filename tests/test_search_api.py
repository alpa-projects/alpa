"""Test the search API."""

import unittest
from functools import partial

import jax
import jax.numpy as jnp
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
import numpy as np
from flax import linen as nn
from flax import optim
import ray

import parax
from parax import parallelize, global_config, testing, PhysicalDeviceMesh, DeviceCluster
from parax.testing import assert_only_has_allreduce


class SearchAPITest(unittest.TestCase):
    def setUp(self):
        #ray.init(address="auto")
        pass

    def test_search_single_host(self):
        batch_size = 16
        hidden_dim = 128

        class Model(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=hidden_dim)(x)
                x = nn.Dense(features=hidden_dim)(x)
                return x

        @parallelize
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"]) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        x = jnp.ones((batch_size, hidden_dim))
        y = jnp.ones((batch_size, hidden_dim))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # Set parallleize option
        parax.set_parallelize_options(
            devices=jax.devices(),
            enable_mesh_shape_search=True,
            mesh_shape_search_mode="measurement",
        )

        optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SearchAPITest("test_search_single_host"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

