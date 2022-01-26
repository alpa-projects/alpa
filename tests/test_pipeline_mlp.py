import unittest
import os

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import ray

from alpa import (parallelize, set_parallelize_options, mark_pipeline,
                  DeviceCluster, manual_layer_construction)
from alpa.model.model_util import TrainState
from alpa.testing import MLPModel, assert_allclose
from alpa.util import get_ray_namespace_str


class PipelineMLPTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        assert len(jax.local_devices()) >= 4

        ray.init(address="auto",
                 namespace=get_ray_namespace_str(prefix="alpa-unittest"))
        device_cluster = DeviceCluster()
        self.devices = device_cluster.get_virtual_physical_mesh()

    def tearDown(self):
        ray.shutdown()

    def train_2_layer_mlp(self, devices, strategy):
        set_parallelize_options(devices=devices, strategy=strategy)

        def train_step(state, batch):

            def loss_func(params, x, y):
                out = state.apply_fn(params, x)
                loss = jnp.mean((out - y)**2)
                mark_pipeline(name="2", mark_type="end")
                return loss

            loss_func = manual_layer_construction(loss_func)
            grads = jax.grad(loss_func)(state.params, batch["x"], batch["y"])
            return grads

        batch_size = 64
        hidden_dim = 1024
        input_dim = output_dim = hidden_dim

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = MLPModel(hidden_dim=hidden_dim, output_dim=output_dim)
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
        pipelined_train_step = parallelize(donate_argnums=())(train_step)
        gradients_with_pipeline = pipelined_train_step(state, batch)

        # Check results
        assert_allclose(gradients, gradients_with_pipeline)
        pipelined_train_step.get_executable(state, batch).shutdown()

    def test_2_layer_mlp_local_pipeline_parallel(self):
        self.train_2_layer_mlp(self.devices, "local_pipeline_parallel")

    @unittest.skip("This test is failing because it's not using apply grad")
    def test_2_layer_mlp_3d_parallel(self):
        self.train_2_layer_mlp(self.devices, "3d_parallel")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineMLPTest("test_2_layer_mlp_local_pipeline_parallel"))
    suite.addTest(PipelineMLPTest("test_2_layer_mlp_3d_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
