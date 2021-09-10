"""Test auto sharding with simple computational graphs."""

import os
import unittest

import numpy as np

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import ray

from parax import (parallelize, set_parallelize_options, grad, testing,
                   global_config, PhysicalDeviceMesh, DeviceCluster)
from parax.testing import assert_allclose


class GradAccumulationTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        ray.init(address="auto", ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def run_gradient_accumulation(self, use_ray, use_2d_mesh):
        if use_ray:
            device_cluster = DeviceCluster()
            physical_mesh = device_cluster.get_physical_mesh()
            logical_mesh = physical_mesh.get_default_logical_mesh()
        else:
            physical_mesh = PhysicalDeviceMesh(jax.local_devices()[:4])
            if use_2d_mesh:
                logical_mesh = physical_mesh.get_logical_mesh([2, 2], [1, 1], [1, 1])
            else:
                logical_mesh = physical_mesh.get_logical_mesh([1, 4], [1, 1], [1, 1])

        set_parallelize_options(logical_mesh)

        global_config.allow_all_to_all = False

        batch_size = 256
        num_micro_batches = 2
        hidden_size = 16
        use_bias = True

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(hidden_size, use_bias=use_bias)(x)
                x = nn.Dense(hidden_size, use_bias=use_bias)(x)
                return x

        batch = {
            "x": jnp.ones(
                (batch_size, hidden_size)) * jnp.arange(batch_size)[:, None],
            "y": jnp.ones((batch_size, hidden_size)),
        }

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, batch["x"])
        optimizer = optim.Momentum(1e-2).create(params)

        def train_step(optimizer, batch, apply_func):

            def loss_func(params):
                out = apply_func(params, batch['x'])
                return jnp.mean((out - batch['y'])**2)

            grads = grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grads)
            return new_optimizer

        # Serial execution
        optimizer_expected = train_step(optimizer, batch, model.apply)

        # Distributed execution
        global_config.num_micro_batches = num_micro_batches
        train_step_parallel = parallelize(train_step)
        optimizer_actual = train_step_parallel(optimizer, batch, model.apply)

        # Check results
        assert_allclose(optimizer_expected.target, optimizer_actual.target)

        physical_mesh.shutdown()

    def test_gradient_accumulation_single_host(self):
        self.run_gradient_accumulation(use_ray=False, use_2d_mesh=False)

    def test_gradient_accumulation_multi_host(self):
        self.run_gradient_accumulation(use_ray=True, use_2d_mesh=False)

    def test_gradient_accumulation_single_host_2d_mesh(self):
        self.run_gradient_accumulation(use_ray=False, use_2d_mesh=True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        GradAccumulationTest("test_gradient_accumulation_single_host"))
    suite.addTest(GradAccumulationTest("test_gradient_accumulation_multi_host"))
    suite.addTest(
        GradAccumulationTest("test_gradient_accumulation_single_host_2d_mesh"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
