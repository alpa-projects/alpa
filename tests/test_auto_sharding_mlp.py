"""Test auto sharding with MLP."""

import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import optim
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis

from parax import parallelize, set_parallelize_options, testing, PhysicalDeviceMesh
from parax.global_env import global_config
from parax.testing import assert_only_has_allreduce,\
    assert_only_has_reduce_scatter_all_gather, assert_has_reduce_scatter
from parax.util import map_to_shape

MB = 1024 ** 2


def assert_close(x, y):
    assert abs(x / y - 1) < 0.01, f"{x} vs. {y}"


def assert_less_equal(x, y):
    assert abs(x / y) < 1.01, f"{x} vs. {y}"


def all_reduce_cost(num_devices, num_bytes):
    return 2.0 * (num_devices - 1) / num_devices * num_bytes


def assert_column_partitioned(x, num_chunks, mesh_dim):
    assert x.sharding_spec.sharding == (NoSharding(), Chunked([num_chunks]))
    assert x.sharding_spec.mesh_mapping[mesh_dim] == ShardedAxis(0)


def assert_row_partitioned(x, num_chunks, mesh_dim):
    assert x.sharding_spec.sharding == (Chunked([num_chunks]), NoSharding())
    assert x.sharding_spec.mesh_mapping[mesh_dim] == ShardedAxis(0)


def assert_replicated_column_partitioned(x, mesh_shape):
    assert x.sharding_spec.sharding == (NoSharding(), Chunked([mesh_shape[1]]))
    assert x.sharding_spec.mesh_mapping[0] == Replicated(mesh_shape[0])
    assert x.sharding_spec.mesh_mapping[1] == ShardedAxis(0)


def assert_replicated_row_partitioned(x, mesh_shape):
    assert x.sharding_spec.sharding == (Chunked([mesh_shape[1]]), NoSharding())
    assert x.sharding_spec.mesh_mapping[0] == Replicated(mesh_shape[0])
    assert x.sharding_spec.mesh_mapping[1] == ShardedAxis(0)


def assert_all_replicated(x, num_replicas):
    for axis_shard in x.sharding_spec.sharding:
        assert axis_shard == NoSharding()
    assert x.sharding_spec.mesh_mapping[0] == Replicated(num_replicas)


def assert_sharded(x):
    for axis in x.sharding_spec.mesh_mapping:
        if isinstance(axis, ShardedAxis):
            return
    assert False, f"Not sharded: {str(x.sharding_spec)}"


def assert_fully_sharded(x):
    for axis in x.sharding_spec.mesh_mapping:
        if not isinstance(axis, ShardedAxis):
            assert False, f"Not fully sharded: {str(x.sharding_spec)}"


class AutoShardingMLPTest(unittest.TestCase):
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

    def run_n_layer_mlp(self, num_layers, batch_size,
                        input_dim, output_dim, hidden_dim, device_mesh):
        set_parallelize_options(devices=device_mesh)

        class Model(nn.Module):
            @nn.compact
            def __call__(self, x):
                for i in range(num_layers - 1):
                    x = nn.Dense(features=hidden_dim)(x)
                    x = nn.relu(x)
                x = nn.Dense(features=output_dim)(x)
                return x

        @parallelize
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"]) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

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

    def test_n_layer_mlp_data_parallel(self):
        num_layers = 6
        batch_size = 512
        hidden_dim = 64

        # Test on different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_n_layer_mlp(
              num_layers, batch_size, hidden_dim, hidden_dim, hidden_dim, device_mesh
            )

            # Check communication cost
            expected = num_layers * (
                device_mesh.all_reduce_cost(hidden_dim * hidden_dim * 4, i) +
                device_mesh.all_reduce_cost(hidden_dim * 4, i))
            assert_close(objective, expected)

            if global_config.prefer_reduce_scatter:
                assert_has_reduce_scatter(hlo_ir)
            else:
                assert_only_has_allreduce(hlo_ir)

            # Check sharding specification
            for weight in jax.tree_util.tree_leaves(optimizer.target):
                assert_all_replicated(weight, np.prod(mesh_shape))

            if global_config.prefer_reduce_scatter:
                for weight in jax.tree_util.tree_leaves(optimizer.state.param_states):
                    if len(weight.shape) > 1:
                        assert_sharded(weight)

    def test_n_layer_mlp_model_parallel(self):
        num_layers = 6
        batch_size = 64
        hidden_dim = 512

        # Test on different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_n_layer_mlp(
              num_layers, batch_size, hidden_dim, hidden_dim, hidden_dim, device_mesh
            )

            # Check communication cost
            expected = (num_layers - 1) *\
              device_mesh.all_reduce_cost(batch_size * hidden_dim * 4, i)
            assert_close(objective, expected)

            if not global_config.prefer_reduce_scatter:
                assert_only_has_allreduce(hlo_ir)

            # Check sharding specification
            for k in range(num_layers):
                weight = optimizer.target["params"][f"Dense_{k}"]["kernel"]
                if k % 2 == 0:
                    assert_column_partitioned(weight, mesh_shape[i], i)
                else:
                    assert_row_partitioned(weight, mesh_shape[i], i)

    def test_n_layer_mlp_2d_mesh(self):
        num_layers = 2
        batch_size = 512
        hidden_dim = 64

        # Test on different device meshes
        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 0.1])
        optimizer, hlo_ir, objective = self.run_n_layer_mlp(
          num_layers, batch_size, hidden_dim, hidden_dim, hidden_dim, device_mesh
        )

        # Check communication cost
        expected = num_layers * (
            device_mesh.all_reduce_cost(hidden_dim * hidden_dim * 4 / mesh_shape[1], 0) +
            device_mesh.all_reduce_cost(hidden_dim * 4, 0)) +\
            (num_layers - 1) *\
            device_mesh.all_reduce_cost(batch_size * hidden_dim * 4 / mesh_shape[0], 1)
        assert_close(objective, expected)

        if global_config.prefer_reduce_scatter:
            assert_has_reduce_scatter(hlo_ir)
        else:
            assert_only_has_allreduce(hlo_ir)

        # Check sharding specification
        for k in range(num_layers):
            weight = optimizer.target["params"][f"Dense_{k}"]["kernel"]
            if k % 2 == 0:
                assert_replicated_column_partitioned(weight, mesh_shape)
            else:
                assert_replicated_row_partitioned(weight, mesh_shape)

        if global_config.prefer_reduce_scatter:
            for weight in jax.tree_util.tree_leaves(optimizer.state.param_states):
                if len(weight.shape) > 1:
                    assert_fully_sharded(weight)

    def test_n_layer_mlp_data_parallel_reduce_scatter(self):
        global_config.prefer_reduce_scatter = True
        self.test_n_layer_mlp_data_parallel()

    def test_n_layer_mlp_model_parallel_reduce_scatter(self):
        global_config.prefer_reduce_scatter = True
        self.test_n_layer_mlp_model_parallel()

    def test_n_layer_mlp_2d_mesh_reduce_scatter(self):
        global_config.prefer_reduce_scatter = True
        self.test_n_layer_mlp_2d_mesh()

    def test_weight_init(self):
        set_parallelize_options(devices=self.devices)

        class Model(nn.Module):
            @nn.compact
            def __call__(self, x, deterministic):
                x = nn.Dense(16)(x)
                x = nn.Dense(16)(x)
                return x

        x = jnp.ones((64, 16))
        y = jnp.ones((64, 16))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)

        @parallelize
        def init_weight(rngkey):
            params = model.init(rngkey, x, True)
            optimizer = optim.Adam(1e-2).create(params)
            return optimizer

        optimizer = init_weight(rngkey)

        # Check sharding specification
        assert_all_replicated(optimizer.state.step, len(self.devices))
        assert_sharded(optimizer.target["params"]["Dense_0"]["kernel"])
        assert_sharded(optimizer.target["params"]["Dense_1"]["kernel"])
        assert_sharded(optimizer.state.param_states["params"]["Dense_0"]["kernel"].grad_ema)
        assert_sharded(optimizer.state.param_states["params"]["Dense_1"]["kernel"].grad_sq_ema)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMLPTest("test_n_layer_mlp_data_parallel"))
    suite.addTest(AutoShardingMLPTest("test_n_layer_mlp_model_parallel"))
    suite.addTest(AutoShardingMLPTest("test_n_layer_mlp_2d_mesh"))

    suite.addTest(AutoShardingMLPTest("test_n_layer_mlp_data_parallel_reduce_scatter"))
    suite.addTest(AutoShardingMLPTest("test_n_layer_mlp_model_parallel_reduce_scatter"))
    suite.addTest(AutoShardingMLPTest("test_n_layer_mlp_2d_mesh_reduce_scatter"))

    suite.addTest(AutoShardingMLPTest("test_weight_init"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

