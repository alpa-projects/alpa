"""Test auto sharding with MLP."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import optim
from flax.training.train_state import TrainState
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
import optax

from parax import parallelize, set_parallelize_options, testing, PhysicalDeviceMesh
from parax.global_env import global_config
from parax.util import map_to_shape, count_communication_primitives

class AutoShardingMixedTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = jax.local_devices()[:4]

        # Backup global config
        self.old_global_config = global_config.backup()

    def tearDown(self):
        # Restore global config
        global_config.restore(self.old_global_config)

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        device_mesh = PhysicalDeviceMesh(devices=self.devices[:np.prod(shape)])
        return device_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def test_dot_all_to_all(self):
        device_mesh = self.get_device_mesh([2, 2], [1, 1], [1, 1])
        set_parallelize_options(devices=device_mesh)

        use_bias = False

        batch_size = 128
        in_dim = 8
        out_dim = in_dim * 32

        global_config.allow_all_gather = False

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=in_dim, use_bias=use_bias)(x)       # R
                x = nn.Dense(features=in_dim * 2, use_bias=use_bias)(x)   # R
                x = nn.Dense(features=in_dim * 4, use_bias=use_bias)(x)   # R
                x = nn.Dense(features=in_dim * 8, use_bias=use_bias)(x)   # S1
                x = nn.Dense(features=in_dim * 16, use_bias=use_bias)(x)  # S0
                x = nn.Dense(features=in_dim * 32, use_bias=use_bias)(x)  # S1
                return x

        @parallelize
        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = jax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        x = jnp.ones((batch_size, in_dim))
        y = jnp.ones((batch_size, out_dim))

        # Init train state
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        # JIT compile
        state = train_step(state, {"x": x, "y": y})

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMixedTest("test_dot_all_to_all"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
