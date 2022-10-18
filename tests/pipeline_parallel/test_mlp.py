import unittest
import os

import jax
import jax.numpy as jnp
import optax
import ray

from alpa import init, parallelize, PipeshardParallel, get_global_physical_mesh
from alpa.model.model_util import TrainState
from alpa.parallel_method import LocalPipelineParallel
from alpa.pipeline_parallel.layer_construction import manual_layer_construction
from alpa.testing import MLPModel, assert_allclose


class PipelineMLPTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    def train_2_layer_mlp(self, method):

        def train_step(state, batch):

            @manual_layer_construction
            def loss_func(params, x, y):
                out = state.apply_fn(params, x)
                # test constant handling
                out = out + jnp.array(range(batch_size)).reshape((-1, 1))
                loss = jnp.mean((out - y)**2)
                return loss

            # Note, we can only use jax.grad in this testcase.
            # TODO: Fix https://github.com/alpa-projects/alpa/issues/560
            grads = jax.grad(loss_func)(state.params, batch["x"], batch["y"])
            return grads

        batch_size = 64
        hidden_size = 1024

        x = jnp.ones((batch_size, hidden_size))
        y = jnp.ones((batch_size, hidden_size))

        # Init model and optimizer
        model = MLPModel(num_layers=4,
                         hidden_size=hidden_size,
                         add_manual_pipeline_marker=True)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # Train step
        batch = {"x": x, "y": y}
        gradients = train_step(state, batch)
        p_train_step = parallelize(train_step, donate_argnums=(), method=method)
        gradients_with_pipeline = p_train_step(state, batch)
        
        # All of the below print -1.
        # mesh = get_global_physical_mesh(create_if_not_exist=True)
        # print(mesh.get_memory_allocated())
        # print(mesh.get_max_memory_allocated())
        # print(mesh.get_available_memory())
        
        # cpu = jax.local_devices(backend="cpu")[0]
        # print(cpu.memory_allocated())
        # print(cpu.max_memory_allocated())
        # print(cpu.available_memory())
        # gpu = jax.local_devices()[0]
        # print(gpu.memory_allocated())
        # print(gpu.max_memory_allocated())
        # print(gpu.available_memory())

        # if isinstance(method, LocalPipelineParallel):
            # jax.profiler.save_device_memory_profile(
                # f"/home/dlzou/projects/alpa-experiments/swap_{method.swap}.prof")

        # Check results
        assert_allclose(gradients, gradients_with_pipeline)

        # Check debug utilities
        if isinstance(method, PipeshardParallel):
            executable = p_train_step.get_last_executable()
            executable.dump_debug_info("tmp")

    def test_2_layer_mlp_local_pipeline_parallel(self):
        # init(cluster="local")
        self.train_2_layer_mlp(LocalPipelineParallel(swap=True))

    def test_2_layer_mlp_pipeshard_parallel(self):
        init(cluster="ray")
        self.train_2_layer_mlp(PipeshardParallel(layer_option="manual"))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineMLPTest("test_2_layer_mlp_local_pipeline_parallel"))
    suite.addTest(PipelineMLPTest("test_2_layer_mlp_pipeshard_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
