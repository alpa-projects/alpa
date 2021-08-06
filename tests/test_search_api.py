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


def get_number_of_lines(filename):
    ct = 0
    for _ in open(filename):
        ct += 1
    return ct


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

    def test_measurement_record(self):
        filename = "tmp.json"
        parax.set_parallelize_options(
            devices=jax.devices(),
            search_logical_mesh_shape=True,
            mesh_shape_search_mode="measurement",
            mesh_shape_search_log_file=filename,
        )

        if os.path.exists(filename):
            os.remove(filename)

        # Run search and dump results into the file
        self.run_2_layer_mlp(batch_size=16, hidden_dim=64)
        before = get_number_of_lines(filename)

        # Load the results without search
        self.run_2_layer_mlp(batch_size=16, hidden_dim=64)
        after = get_number_of_lines(filename)

        # The second call should not generate new records
        assert before == after


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SearchAPITest("test_search_single_host"))
    suite.addTest(SearchAPITest("test_search_multi_host"))
    suite.addTest(SearchAPITest("test_measurement_record"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

