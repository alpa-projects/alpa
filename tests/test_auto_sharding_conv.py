"""Test auto sharding with convoluton nets."""

import unittest
from typing import Any

from flax import linen as nn, optim
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from parax import parallelize, set_parallelize_options, testing, PhysicalDeviceMesh
from parax.global_env import global_config
from parax.util import map_to_shape, count_communication_primitives

from test_auto_sharding_mlp import assert_close, assert_all_replicated


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: optim.DynamicScale


def assert_data_parallel_cost(state,
                              hlo_ir,
                              objective,
                              device_mesh,
                              mesh_dim,
                              allow_not_sharded_params=0):
    params = jax.tree_util.tree_leaves(state.params)
    opt_state = jax.tree_util.tree_leaves(state.opt_state)
    batch_stats = jax.tree_util.tree_leaves(state.batch_stats)

    # Check communication cost
    replicated_penalty = int(
        device_mesh.all_reduce_cost(1, 0) + device_mesh.all_reduce_cost(1, 1))
    weight_sync = sum(
        device_mesh.all_reduce_cost(np.prod(x.shape) * 4, mesh_dim) +
        replicated_penalty for x in params)
    num_batch_norm = len(batch_stats) // 2
    batch_norm_sync = 2 * sum(
        device_mesh.all_reduce_cost(np.prod(x.shape) * 4, mesh_dim) +
        replicated_penalty for x in batch_stats)
    expected = weight_sync + batch_norm_sync

    assert_close(objective, expected, atol=0.05)

    # Check number of communication primitives
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ =\
        count_communication_primitives(hlo_ir, ignore_scalar_all_reduce=True)
    assert n_all_reduce == 1 + num_batch_norm * 2
    assert n_total == n_all_reduce

    for weight in params:
        assert_all_replicated(weight, np.prod(device_mesh.id_mesh.shape))


class AutoShardingConvTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = jax.local_devices()[:4]

        # Backup global config
        self.old_global_config = global_config.backup()

    def tearDown(self):
        # Restore global config
        global_config.restore(self.old_global_config)

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        device_mesh = PhysicalDeviceMesh(devices=self.devices)
        return device_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_n_layer_conv(self,
                         num_layers,
                         batch_size,
                         image_size,
                         channel,
                         device_mesh,
                         use_bias=False):
        set_parallelize_options(devices=device_mesh)

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x, train=True):
                for i in range(num_layers):
                    x = nn.Conv(features=channel,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                use_bias=use_bias)(x)
                    x = nn.BatchNorm(use_running_average=not train)(x)
                    x = nn.relu(x)
                    x = nn.max_pool(x,
                                    window_shape=(2, 2),
                                    strides=(1, 1),
                                    padding="SAME")
                return x

        @parallelize
        def train_step(state, batch):

            def loss_func(params):
                out, new_model_state = state.apply_fn(
                    {
                        "params": params,
                        "batch_stats": state.batch_stats
                    },
                    batch["x"],
                    mutable=['batch_stats'])
                loss = jnp.mean((out - batch["y"])**2)
                return loss, new_model_state

            grads, new_model_state = jax.grad(loss_func,
                                              has_aux=True)(state.params)
            new_state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state['batch_stats'])
            return new_state

        x = jnp.ones((batch_size, image_size, image_size, channel))
        out_image_size = image_size // (2**num_layers)
        y = jnp.ones((batch_size, out_image_size, out_image_size, channel))

        # Init train state
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.sgd(0.1)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params["params"],
                                  tx=tx,
                                  batch_stats=params["batch_stats"],
                                  dynamic_scale=None)

        # JIT compile
        state = train_step(state, {"x": x, "y": y})

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()
        return state, hlo_ir, testing.last_compiled_auto_sharding_objective

    def test_n_layer_conv_data_parallel(self):
        batch_size = 16
        image_size = 64
        num_layers = 3
        channel = 4

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_n_layer_conv(
                num_layers, batch_size, image_size, channel, device_mesh)

            assert_data_parallel_cost(state, hlo_ir, objective, device_mesh, i)

    def test_n_layer_conv_model_parallel(self):
        batch_size = 8
        image_size = 16
        num_layers = 4
        channel = 1024

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_n_layer_conv(
                num_layers, batch_size, image_size, channel, device_mesh)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ =\
                count_communication_primitives(hlo_ir, ignore_scalar_all_reduce=True)

            assert n_all_reduce == num_layers - 1
            assert n_total == n_all_reduce

    def test_n_layer_conv_2d_mesh(self):
        batch_size = 16
        image_size = 64
        num_layers = 4
        channel = 8

        # Test on different device meshes
        device_mesh = self.get_device_mesh([2, 2], [1, 1], [1, 0.1])
        state, hlo_ir, objective = self.run_n_layer_conv(
            num_layers, batch_size, image_size, channel, device_mesh)

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ =\
            count_communication_primitives(hlo_ir, ignore_scalar_all_reduce=True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingConvTest("test_n_layer_conv_data_parallel"))
    suite.addTest(AutoShardingConvTest("test_n_layer_conv_model_parallel"))
    suite.addTest(AutoShardingConvTest("test_n_layer_conv_2d_mesh"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
