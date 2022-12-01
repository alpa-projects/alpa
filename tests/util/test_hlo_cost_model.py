"""Test HLO cost model."""
import pickle
import unittest

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import ray

from alpa import (init, parallelize, global_config, ShardParallel,
                  LocalPhysicalDeviceMesh, ProfilingResultDatabase)
from alpa.device_mesh import get_global_cluster
from alpa.mesh_profiling import estimate_hlo_module_cost
from alpa.util import map_to_shape


class HloCostModelTest(unittest.TestCase):

    def run_n_layer_mlp(self,
                        num_layers,
                        batch_size,
                        input_dim,
                        output_dim,
                        hidden_dim,
                        device_mesh,
                        use_bias=True):

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                for i in range(num_layers - 1):
                    x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                    x = nn.relu(x)
                x = nn.Dense(features=output_dim, use_bias=use_bias)(x)
                return x

        @parallelize(method=ShardParallel(devices=device_mesh))
        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = jax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init train state
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.adam(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        # Get optimized HLO IR
        executable = train_step.get_executable(state, {"x": x, "y": y})
        return executable.compiled.hlo_modules()[0]

    def test_cluster_profling(self):
        init(cluster="ray")
        cluster = get_global_cluster()
        manually_specified_submeshes = [
            (1, 1),
            cluster.get_virtual_physical_mesh().shape,
        ]

        prof_database = cluster.profile_all(
            "p3.16",
            2,
            2,
            max_fail_retry=5,
            cache_filename="tmp_cache.pkl",
            dot_range=(0, 1),
            mesh_size_choices=manually_specified_submeshes)
        prof_database.save("tmp_prof_database.pkl")

    @unittest.skip("Temporary disabled due to being flaky")
    def test_n_layer_mlp(self):
        num_layers = 2
        batch_size = 32
        hidden_dim = 16

        prof_database = ProfilingResultDatabase()
        prof_database.load("tmp_prof_database.pkl")

        device_mesh = LocalPhysicalDeviceMesh()
        hlo_module = self.run_n_layer_mlp(num_layers, batch_size, hidden_dim,
                                          hidden_dim, hidden_dim, device_mesh)
        mesh_result = prof_database.query("p3.16", device_mesh.shape)
        cost = estimate_hlo_module_cost(hlo_module, mesh_result)
        # assert cost > 0


def suite():
    suite = unittest.TestSuite()
    suite.addTest(HloCostModelTest("test_cluster_profling"))
    suite.addTest(HloCostModelTest("test_n_layer_mlp"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
