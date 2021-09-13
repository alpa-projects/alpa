"""Test auto sharding with convoluton nets."""

import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import optim

from parax import parallelize, set_parallelize_options, testing, PhysicalDeviceMesh
from parax.global_env import global_config
from parax.util import map_to_shape, count_communication_primitives

from test_auto_sharding_mlp import assert_data_parallel_cost


class AutoShardingConvTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = jax.local_devices()[:4]

        # Backup global config
        self.old_global_config = copy.deepcopy(global_config.__dict__)

    def tearDown(self):
        # Restore global config
        global_config.__dict__ = self.old_global_config

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        device_mesh = PhysicalDeviceMesh(devices=self.devices)
        return device_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_n_layer_conv(self,
                         num_layers,
                         batch_size,
                         image_size,
                         channel,
                         device_mesh,
                         use_bias=True):
        set_parallelize_options(devices=device_mesh)

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                for i in range(num_layers - 1):
                    x = nn.Conv(features=channel, kernel_size=(3, 3),
                                strides=(2, 2), use_bias=use_bias)(x)
                    x = nn.relu(x)
                x = nn.Conv(features=channel, kernel_size=(3, 3),
                            strides=(2, 2), use_bias=use_bias)(x)
                return x

        @parallelize
        def train_step(optimizer, batch, apply_fn):

            def loss_func(params):
                out = apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        x = jnp.ones((batch_size, image_size, image_size, channel))
        out_image_size = image_size // (2 ** num_layers)
        y = jnp.ones((batch_size, out_image_size, out_image_size, channel))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.Adam(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()
        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def test_n_layer_conv_data_parallel(self):
        batch_size = 32
        image_size = 32
        num_layers = 1
        channel = 4

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_n_layer_conv(
                num_layers, batch_size, image_size, channel, device_mesh)

            assert_data_parallel_cost(optimizer, hlo_ir, objective, device_mesh, i)
            
            return


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingConvTest("test_n_layer_conv_data_parallel"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

