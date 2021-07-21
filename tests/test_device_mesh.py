"""Test distributed mulit-host device mesh."""

import os
import unittest

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import numpy as np
import ray

from parax import DeviceCluster, parallelize, set_parallelize_options, testing
from parax.testing import assert_allclose


class DeviceMeshTest(unittest.TestCase):
    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        ray.init(address="auto", ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def test_add_one(self):
        # Launch a multi-host device mesh
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        total_devices = len(physical_mesh.host_ids) * physical_mesh.num_devices_per_host
        logical_mesh = physical_mesh.get_logical_mesh([1, total_devices])
        set_parallelize_options(devices=logical_mesh)

        @parallelize
        def add_one(x):
            x = x + 1
            return x

        @parallelize
        def multiply_two(x):
            x = x * 2
            return x

        # Run computation
        a = jnp.ones((512, 512))
        out = add_one(a)
        out = multiply_two(out)

        # Check results
        assert_allclose(out._value, (np.ones_like(a) + 1) * 2)

        physical_mesh.shutdown()

    def test_mlp(self):
        # Launch a multi-host device mesh
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        set_parallelize_options(devices=physical_mesh)

        batch_size = 32
        input_dim = hidden_dim = output_dim = 32

        class Model(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=hidden_dim)(x)
                x = nn.relu(x)
                x = nn.Dense(features=output_dim)(x)
                return x

        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"]) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer


        # One batch of data and label
        batch = {
            "x": np.random.randn(batch_size, input_dim),
            "y": np.random.randn(batch_size, output_dim),
        }

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, batch["x"])
        optimizer = optim.GradientDescent(1e-2).create(params)

        # Serial execution
        optimizer_expected = train_step(optimizer, batch, model.apply)

        # Distributed execution
        train_step_parallel = parallelize(train_step)
        optimizer_actual = train_step_parallel(optimizer, batch, model.apply)

        # Check results
        assert_allclose(optimizer_expected.target, optimizer_actual.target)

        physical_mesh.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DeviceMeshTest("test_add_one"))
    suite.addTest(DeviceMeshTest("test_mlp"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

