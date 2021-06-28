"""Test the search API."""

from functools import partial
import os
import unittest

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
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        ray.init(address="auto", ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def run_2_layer_mlp(self, batch_size, hidden_dim):
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

        # Compile and run
        optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

    def test_search_single_host(self):
        parax.set_parallelize_options(
            devices=jax.devices(),
            search_logical_mesh_shape=True,
            mesh_shape_search_mode="measurement",
        )

        self.run_2_layer_mlp(batch_size=16, hidden_dim=64)

    def test_search_multi_host(self):
        physical_mesh = DeviceCluster().get_physical_mesh()

        parax.set_parallelize_options(
            devices=physical_mesh,
            search_logical_mesh_shape=True,
            mesh_shape_search_mode="measurement",
        )

        self.run_2_layer_mlp(batch_size=16, hidden_dim=64)

        physical_mesh.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SearchAPITest("test_search_single_host"))
    suite.addTest(SearchAPITest("test_search_multi_host"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

