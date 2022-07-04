# pylint: disable=missing-class-docstring
"""Some basic tests to test installation."""
import os
import unittest

from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from alpa import (init, parallelize, grad, ShardParallel, PipeshardParallel,
                  AutoLayerOption)
from alpa.device_mesh import get_global_cluster
from alpa.testing import assert_allclose


def create_train_state_and_batch(batch_size, hidden_size):

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
        "x":
            jax.random.normal(rngkey, (batch_size, hidden_size),
                              dtype=jnp.float32),
        "y":
            jax.random.normal(rngkey, (batch_size, hidden_size),
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
        state, batch = create_train_state_and_batch(256, 256)

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        # Serial execution
        expected_state = train_step(state, batch)

        # Parallel execution
        p_train_step = parallelize(train_step,
                                   method=ShardParallel(num_micro_batches=2))
        actual_state = p_train_step(state, batch)

        # Check results
        assert_allclose(expected_state.params, actual_state.params)

    def test_2_pipeline_parallel(self):
        init(cluster="ray")

        layer_num = min(get_global_cluster().num_devices, 2)
        state, batch = create_train_state_and_batch(256, 256)

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        # Serial execution
        expected_state = train_step(state, batch)

        # Parallel execution
        p_train_step = parallelize(
            train_step,
            method=PipeshardParallel(
                num_micro_batches=2,
                layer_option=AutoLayerOption(layer_num=layer_num)))
        actual_state = p_train_step(state, batch)

        # Check results
        assert_allclose(expected_state.params, actual_state.params)


def suite():
    s = unittest.TestSuite()
    s.addTest(InstallationTest("test_1_shard_parallel"))
    s.addTest(InstallationTest("test_2_pipeline_parallel"))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
