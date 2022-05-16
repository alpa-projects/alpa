"""Test auto sharding with mixed mesh shape."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
import optax

from alpa import parallelize, LocalPhysicalDeviceMesh, ShardParallel, AutoShardingOption
from alpa.util import map_to_shape, count_communication_primitives


class AutoShardingMixedTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:4])

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def test_dot_all_to_all(self):
        device_mesh = self.get_device_mesh([2, 2], [1, 1], [1, 0.1])

        as_option = AutoShardingOption(allow_mixed_mesh_shape=True,
                                       allow_all_gather=False)

        use_bias = False
        B = 256
        E = 4
        M = 16
        M_ = M // E
        H = M * 8

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                wi = self.param("wi", jax.nn.initializers.zeros, (
                    E,
                    M_,
                    H,
                ))
                wo = self.param("wo", jax.nn.initializers.zeros, (
                    E,
                    H,
                    M_,
                ))

                x = nn.Dense(features=M, use_bias=use_bias)(x)
                x = nn.Dense(features=M, use_bias=use_bias)(x)
                x = x.reshape((B, E, M_))

                x = jnp.einsum("BEM,EMH->EBH", x, wi)
                x = jnp.einsum("EBH,EHM->BEM", x, wo)

                x = x.reshape((B, M))
                x = nn.Dense(features=M, use_bias=use_bias)(x)
                x = nn.Dense(features=M, use_bias=use_bias)(x)
                return x

        @parallelize(method=ShardParallel(devices=device_mesh,
                                          auto_sharding_option=as_option))
        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = jax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        x = jnp.ones((B, M))
        y = jnp.ones((B, M))

        # Init train state
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        # JIT compile
        executable = train_step.get_executable(state, {"x": x, "y": y})
        hlo_ir = executable.get_hlo_text()

        # Check sharding specs
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all = (
            count_communication_primitives(hlo_ir))
        assert n_all_to_all > 0
        assert n_total == n_all_reduce + n_all_to_all


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMixedTest("test_dot_all_to_all"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
