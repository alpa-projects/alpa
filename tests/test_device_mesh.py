"""Test distributed mulit-host device mesh."""

import os
import unittest

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import numpy as np
import ray

from alpa import DeviceCluster, parallelize, set_parallelize_options, testing, DistributedArray
from alpa.testing import assert_allclose


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
        num_devices = len(
            physical_mesh.host_ids) * physical_mesh.num_devices_per_host
        logical_mesh = physical_mesh.get_logical_mesh([1, num_devices])
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

        batch_size = 16
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
                return jnp.mean((out - batch["y"])**2)

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

    def test_distributed_array(self):
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        logical_mesh = physical_mesh.get_default_logical_mesh()

        array = jnp.ones((16, 16))
        sharding_spec = logical_mesh.make_tile_spec(array, [0, 1], [0, 1])
        indices = sharding_spec.indices(array.shape).flatten()
        remote_a = physical_mesh.shard_args([indices], (False,), (array,))
        physical_mesh.shutdown()

    def test_preshard_args(self):
        # Single host
        set_parallelize_options(devices=None)

        @parallelize
        def add_one(x):
            x = x + 1
            return x

        a = jnp.ones((32, 32))
        a, = add_one.preshard_dynamic_args(a)
        assert isinstance(a, pxla.ShardedDeviceArray)

        # Multi host
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        set_parallelize_options(devices=physical_mesh)

        a = jnp.ones((64, 64))
        a, = add_one.preshard_dynamic_args(a)
        assert isinstance(a, DistributedArray)

        physical_mesh.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DeviceMeshTest("test_add_one"))
    suite.addTest(DeviceMeshTest("test_mlp"))
    suite.addTest(DeviceMeshTest("test_distributed_array"))
    suite.addTest(DeviceMeshTest("test_preshard_args"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
