"""Test auto sharding with MLP."""
import numpy as np

import unittest

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
import numpy as np

from parax import parallelize, SingleHostDeviceMesh, global_config, testing

MB = 1024 ** 2

def assert_close(x, y):
    assert abs(x / y - 1) < 0.01, f"{x} vs. {y}"


def all_reduce_cost(num_devices, num_bytes):
    return 2.0 * (num_devices - 1) / num_devices * num_bytes


def map_to_shape(array_pytree):
    return jax.tree_util.tree_map(lambda x: x.shape, array_pytree)


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


class AutoShardingMLPTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = jax.local_devices()[:4]
        global_config.shard_parallel_strategy = "auto_sharding"

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        device_mesh = SingleHostDeviceMesh(self.devices)
        return device_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_2_layer_mlp(self, batch_size, input_dim, output_dim, hidden_dim,
                        device_mesh):
        class Model(nn.Module):
            hidden_dim: int
            output_dim: int

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=self.hidden_dim)(x)
                x = nn.relu(x)
                x = nn.Dense(features=self.output_dim)(x)
                return x

        @parallelize(devices=device_mesh)
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y']) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = Model(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def run_n_layer_mlp(self, num_layers, batch_size,
                        input_dim, output_dim, hidden_dim, device_mesh):
        class Model(nn.Module):
            hidden_dim: int
            output_dim: int
            num_layers: int

            @nn.compact
            def __call__(self, x):
                for i in range(self.num_layers-1):
                    x = nn.Dense(features=self.hidden_dim)(x)
                    x = nn.relu(x)
                x = nn.Dense(features=self.output_dim)(x)
                return x

        @parallelize(devices=device_mesh)
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y']) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = Model(num_layers=num_layers,
                      hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

        # Get optimized HLO IR
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()
        return optimizer, hlo_ir, testing.last_compiled_auto_sharding_objective

    def test_2_layer_mlp_data_parallel(self):
        batch_size = 512
        hidden_dim = 64

        # Test on different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_2_layer_mlp(
              batch_size, hidden_dim, hidden_dim, hidden_dim, device_mesh
            )

            # Check communication cost
            expected = 2 * (
                device_mesh.all_reduce_cost(hidden_dim * hidden_dim * 4, i) +
                device_mesh.all_reduce_cost(hidden_dim * 4, i))

            assert_close(objective, expected)

            # Check sharding specification
            weight0 = optimizer.target["params"]["Dense_0"]["kernel"]
            weight1 = optimizer.target["params"]["Dense_1"]["kernel"]
            assert_all_replicated(weight0, np.prod(mesh_shape))
            assert_all_replicated(weight1, np.prod(mesh_shape))

    def test_2_layer_mlp_model_parallel(self):
        batch_size = 64
        hidden_dim = 512

        # Test on different device meshes
        for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
            device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 1])
            optimizer, hlo_ir, objective = self.run_2_layer_mlp(
              batch_size, hidden_dim, hidden_dim, hidden_dim, device_mesh
            )

            # Check communication cost
            expected = device_mesh.all_reduce_cost(batch_size * hidden_dim * 4, i)
            assert_close(objective, expected)

            # Check sharding specification
            weight0 = optimizer.target["params"]["Dense_0"]["kernel"]
            weight1 = optimizer.target["params"]["Dense_1"]["kernel"]
            assert_column_partitioned(weight0, mesh_shape[i], i)
            assert_row_partitioned(weight1, mesh_shape[i], i)

    def test_2_layer_mlp_2d_mesh(self):
        batch_size = 512
        hidden_dim = 64

        # Test on different device meshes
        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 0.01])
        optimizer, hlo_ir, objective = self.run_2_layer_mlp(
          batch_size, hidden_dim, hidden_dim, hidden_dim, device_mesh
        )

        # Check communication cost
        expected = 2 * (
            device_mesh.all_reduce_cost(hidden_dim * hidden_dim * 4 / mesh_shape[1], 0) +
            device_mesh.all_reduce_cost(hidden_dim * 4, 0)) +\
            device_mesh.all_reduce_cost(batch_size * hidden_dim * 4 / mesh_shape[0], 1)
        assert_close(objective, expected)

        # Check sharding specification
        weight0 = optimizer.target["params"]["Dense_0"]["kernel"]
        weight1 = optimizer.target["params"]["Dense_1"]["kernel"]
        assert_replicated_column_partitioned(weight0, mesh_shape)
        assert_replicated_row_partitioned(weight1, mesh_shape)

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

            # Check sharding specification
            for i in range(num_layers):
                weight = optimizer.target["params"][f"Dense_{i}"]["kernel"]
                assert_all_replicated(weight, np.prod(mesh_shape))

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

            # Check sharding specification
            for k in range(num_layers):
                weight = optimizer.target["params"][f"Dense_{k}"]["kernel"]
                if k % 2 == 0:
                    assert_column_partitioned(weight, mesh_shape[i], i)
                else:
                    assert_row_partitioned(weight, mesh_shape[i], i)

    def test_n_layer_mlp_2d_mesh(self):
        num_layers = 6
        batch_size = 512
        hidden_dim = 64

        # Test on different device meshes
        mesh_shape = [2, 2]
        device_mesh = self.get_device_mesh(mesh_shape, [1, 1], [1, 0.01])
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

        # Check sharding specification
        for k in range(num_layers):
            weight = optimizer.target["params"][f"Dense_{k}"]["kernel"]
            if k % 2 == 0:
                assert_replicated_column_partitioned(weight, mesh_shape)
            else:
                assert_replicated_row_partitioned(weight, mesh_shape)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp_data_parallel'))
    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp_model_parallel'))
    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp_2d_mesh'))

    suite.addTest(AutoShardingMLPTest('test_n_layer_mlp_data_parallel'))
    suite.addTest(AutoShardingMLPTest('test_n_layer_mlp_model_parallel'))
    suite.addTest(AutoShardingMLPTest('test_n_layer_mlp_2d_mesh'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

