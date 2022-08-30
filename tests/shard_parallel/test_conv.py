"""Test auto sharding with convolution nets."""

import unittest
from typing import Any

from flax import linen as nn
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from alpa import parallelize, ShardParallel, LocalPhysicalDeviceMesh, AutoShardingOption
from alpa.util import map_to_shape, count_communication_primitives

from tests.shard_parallel.test_mlp import assert_close, assert_all_replicated, is_sharded


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


def assert_data_parallel_cost(state,
                              hlo_ir,
                              objective,
                              device_mesh,
                              as_option,
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

    # Check numbers of communication primitives
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
        count_communication_primitives(hlo_ir, ignore_scalar_all_reduce=True))

    if as_option.prefer_reduce_scatter:
        assert n_all_reduce == num_batch_norm * 2
        assert n_reduce_scatter > 0
        assert n_all_gather <= 2
        assert n_total == n_all_reduce + n_reduce_scatter + n_all_gather
    else:
        assert n_all_reduce == 1 + num_batch_norm * 2
        assert n_total == n_all_reduce

    if as_option.prefer_reduce_scatter:
        num_not_sharded = 0
        for weight in opt_state:
            if not is_sharded(weight) and len(weight.shape) > 1:
                num_not_sharded += 1
        assert num_not_sharded == 0
    else:
        for weight in params:
            assert_all_replicated(weight, np.prod(device_mesh.shape))


class AutoShardingConvTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.physical_mesh = LocalPhysicalDeviceMesh(jax.local_devices()[:4])
        self.as_option = AutoShardingOption()

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_n_layer_conv(self,
                         num_layers,
                         batch_size,
                         image_size,
                         channel,
                         device_mesh,
                         use_bias=False,
                         is_depthwise=False):
        if not is_depthwise:

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

            x = jnp.ones((batch_size, image_size, image_size, channel))
            out_image_size = image_size // (2**num_layers)
            y = jnp.ones((batch_size, out_image_size, out_image_size, channel))
        else:

            class Model(nn.Module):

                @nn.compact
                def __call__(self, x, train=True):
                    x = nn.Conv(features=8 * channel,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                use_bias=use_bias)(x)
                    x = nn.Conv(features=8 * channel,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                feature_group_count=8 * channel,
                                use_bias=use_bias)(x)
                    x = nn.Conv(features=channel,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                use_bias=use_bias)(x)
                    x = nn.relu(x)
                    x = nn.BatchNorm(use_running_average=not train)(x)
                    return x

            x = jnp.ones((batch_size, image_size, image_size, channel))
            y = jnp.ones((batch_size, image_size, image_size, channel))

        @parallelize(method=ShardParallel(devices=device_mesh,
                                          auto_sharding_option=self.as_option))
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

        # Init train state
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.sgd(0.1, momentum=0.9)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params["params"],
                                  tx=tx,
                                  batch_stats=params["batch_stats"],
                                  dynamic_scale=None)

        # JIT compile
        state = train_step(state, {"x": x, "y": y})

        # Get optimized HLO IR
        executable = train_step.get_last_executable()
        return (state, executable.get_hlo_text(),
                executable.auto_sharding_objective)

    def test_n_layer_conv_data_parallel(self):
        batch_size = 16
        image_size = 16
        num_layers = 3
        channel = 4

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_n_layer_conv(
                num_layers, batch_size, image_size, channel, device_mesh)

            assert_data_parallel_cost(state, hlo_ir, objective, device_mesh,
                                      self.as_option, i)

    def test_n_layer_conv_model_parallel(self):
        batch_size = 8
        image_size = 16
        num_layers = 4
        channel = 256

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_n_layer_conv(
                num_layers, batch_size, image_size, channel, device_mesh)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(hlo_ir,
                                               ignore_scalar_all_reduce=True))

            assert n_all_reduce == num_layers - 1
            assert n_total == n_all_reduce

    def test_n_layer_conv_2d_mesh(self):
        batch_size = 8
        image_size = 32
        num_layers = 4
        channel = 8
        self.as_option.allow_mixed_mesh_shape = False

        device_mesh = self.get_device_mesh([2, 2], [1, 1], [1, 0.1])
        state, hlo_ir, objective = self.run_n_layer_conv(
            num_layers, batch_size, image_size, channel, device_mesh)

        # Check numbers of communication primitives
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all = (
            count_communication_primitives(hlo_ir,
                                           ignore_scalar_all_reduce=True))
        if self.as_option.prefer_reduce_scatter:
            assert n_reduce_scatter > 0
        if self.as_option.allow_mixed_mesh_shape:
            assert n_all_to_all > 0

    def test_n_layer_conv_2d_mesh_mixed_shape(self):
        self.as_option.allow_mixed_mesh_shape = True
        self.test_n_layer_conv_2d_mesh()

    def test_n_layer_conv_data_parallel_reduce_scatter(self):
        self.as_option.prefer_reduce_scatter = True
        self.test_n_layer_conv_data_parallel()

    def test_n_layer_conv_2d_mesh_mixed_shape_reduce_scatter(self):
        self.as_option.allow_mixed_mesh_shape = True
        self.as_option.prefer_reduce_scatter = True
        self.test_n_layer_conv_2d_mesh()

    def test_n_layer_depthwise_conv_model_parallel(self):
        batch_size = 4
        image_size = 8
        num_layers = 2
        channel = 256

        # Test on different device meshes
        for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            state, hlo_ir, objective = self.run_n_layer_conv(num_layers,
                                                             batch_size,
                                                             image_size,
                                                             channel,
                                                             device_mesh,
                                                             is_depthwise=True)

            n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
                count_communication_primitives(hlo_ir,
                                               ignore_scalar_all_reduce=True))
            assert n_all_reduce == 1
            assert n_total == n_all_reduce


def suite():
    suite = unittest.TestSuite()

    def add(name):
        suite.addTest(AutoShardingConvTest(name))

    add("test_n_layer_conv_data_parallel")
    add("test_n_layer_conv_model_parallel")
    add("test_n_layer_conv_2d_mesh")
    add("test_n_layer_conv_2d_mesh_mixed_shape")

    add("test_n_layer_conv_data_parallel_reduce_scatter")
    add("test_n_layer_conv_2d_mesh_mixed_shape_reduce_scatter")

    add("test_n_layer_depthwise_conv_model_parallel")

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
