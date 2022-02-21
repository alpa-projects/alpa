"""Some basic tests to test installation."""

import os
import unittest

from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

from alpa import (parallelize, set_parallelize_options, grad, global_config,
                  automatic_layer_construction, DeviceCluster)
from alpa.testing import assert_allclose


def create_train_state_and_batch():
    batch_size = 16
    num_micro_batches = 2
    hidden_size = 16

    class Model(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(hidden_size, use_bias=True)(x)
            x = nn.Dense(hidden_size, use_bias=True)(x)
            x = nn.Dense(hidden_size, use_bias=True)(x)
            x = nn.Dense(hidden_size, use_bias=True)(x)
            x = nn.Dense(hidden_size, use_bias=True)(x)
            x = nn.Dense(hidden_size, use_bias=True)(x)
            return x

    rngkey = jax.random.PRNGKey(0)
    batch = {
        "x": jax.random.normal(rngkey, (batch_size, hidden_size),
                               dtype=jnp.float32),
        "y": jax.random.normal(rngkey, (batch_size, hidden_size),
                               dtype=jnp.float32)
    }

    # Init model and optimizer
    model = Model()
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, batch["x"])
    tx = optax.sgd(learning_rate=1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    return state, batch


class InstallationTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    def test_1_shard_parallel(self):
        state, batch = create_train_state_and_batch()

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y'])**2)

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        # Serial execution
        expected_state = train_step(state, batch)

        # Parallel execution
        global_config.num_micro_batches = 2
        parallel_train_step = parallelize(train_step)
        actual_state = parallel_train_step(state, batch)

        # Check results
        assert_allclose(expected_state.params, actual_state.params)

    def test_2_pipeline_parallel(self):
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')

        device_mesh = DeviceCluster().get_virtual_physical_mesh()
        set_parallelize_options(devices=device_mesh,
                                strategy="3d_parallel",
                                pipeline_stage_mode="uniform_layer_gpipe")

        state, batch = create_train_state_and_batch()

        def train_step(state, batch):

            @automatic_layer_construction(layer_num=2)
            def loss_func(params):
                out = state.apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y'])**2)

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        # Serial execution
        expected_state = train_step(state, batch)

        # Parallel execution
        global_config.num_micro_batches = 2
        parallel_train_step = parallelize(train_step)
        actual_state = parallel_train_step(state, batch)

        # Check results
        assert_allclose(expected_state.params, actual_state.params)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(InstallationTest("test_1_shard_parallel"))
    suite.addTest(InstallationTest("test_2_pipeline_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
