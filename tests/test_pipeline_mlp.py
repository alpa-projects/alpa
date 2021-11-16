import unittest
import os

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import ray

from parax import (parallelize, set_parallelize_options, mark_pipeline,
                   DeviceCluster, manual_layer_slicing)
from parax.testing import MLPModel, assert_allclose


class PipelineMLPTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        assert len(jax.local_devices()) >= 4

        ray.init(address='auto')
        device_cluster = DeviceCluster()
        mesh = device_cluster.get_virtual_mesh()
        self.devices = mesh

    def tearDown(self):
        ray.shutdown()

    def train_2_layer_mlp(self, devices, strategy):
        set_parallelize_options(devices=devices, strategy=strategy)

        def train_step(optimizer, batch, apply_fn):

            def loss_func(params, x, y):
                out = apply_fn(params, x)
                loss = jnp.mean((out - y)**2)
                mark_pipeline(name='2', mark_type='end')
                return loss

            loss_func = manual_layer_slicing(loss_func)
            grad_param = jax.grad(loss_func)(optimizer.target, batch['x'],
                                             batch['y'])
            return grad_param

        batch_size = 128
        hidden_dim = 2048
        input_dim = output_dim = hidden_dim

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = MLPModel(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # Train step
        gradients = train_step(optimizer, {"x": x, "y": y}, model.apply)
        pipelined_train_step = parallelize(donate_argnums=())(train_step)
        args = (optimizer, {"x": x, "y": y}, model.apply)
        gradients_with_pipeline = pipelined_train_step(*args)

        # Check results
        assert_allclose(gradients, gradients_with_pipeline)
        pipelined_train_step.get_executable(*args).shutdown()

    def test_2_layer_mlp_local_pipeline_parallel(self):
        self.train_2_layer_mlp(self.devices, "local_pipeline_parallel")

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
