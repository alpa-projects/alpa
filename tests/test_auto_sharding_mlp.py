import numpy as np

import unittest

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
import numpy as np

from parax import parallelize, DeviceMesh, global_config, testing

from test_auto_sharding_basic import assert_close, all_reduce_cost

MB = 1024 ** 2


class AutoShardingMLPTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = jax.local_devices()[:4]
        global_config.shard_parallel_strategy = "auto_sharding"

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        devices = np.array(self.devices).reshape(shape)
        return DeviceMesh(devices, mesh_alpha, mesh_beta)

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
            # Weights are replicated
            weight0 = optimizer.target["params"]["Dense_0"]["kernel"]
            weight1 = optimizer.target["params"]["Dense_1"]["kernel"]
            assert weight0.sharding_spec == pxla.ShardingSpec(
                sharding=(NoSharding(), NoSharding()),
                mesh_mapping=(Replicated(np.prod(mesh_shape)),),
            )
            assert weight1.sharding_spec == pxla.ShardingSpec(
                sharding=(NoSharding(), NoSharding()),
                mesh_mapping=(Replicated(np.prod(mesh_shape)),),
            )

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
            # Column partitioned
            assert weight0.sharding_spec == pxla.ShardingSpec(
                sharding=(Chunked([1]), Chunked([4])),
                mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
            )
            # Row partitioned
            assert weight1.sharding_spec == pxla.ShardingSpec(
                sharding=(Chunked([4]), Chunked([1])),
                mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
            )

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
            # Weights are replicated
            for i in range(num_layers):
                weight = optimizer.target["params"][f"Dense_{i}"]["kernel"]
                assert weight.sharding_spec == pxla.ShardingSpec(
                    sharding=(NoSharding(), NoSharding()),
                    mesh_mapping=(Replicated(np.prod(mesh_shape)),),
                )

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
            for i in range(num_layers):
                weight = optimizer.target["params"][f"Dense_{i}"]["kernel"]
                if i % 2 == 0:
                    # Column partitioned
                    assert weight.sharding_spec == pxla.ShardingSpec(
                        sharding=(Chunked([1]), Chunked([4])),
                        mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
                    )
                else:
                    # Row partitioned
                    assert weight.sharding_spec == pxla.ShardingSpec(
                        sharding=(Chunked([4]), Chunked([1])),
                        mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
                    )

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
        # Column partitioned
        assert weight0.sharding_spec == pxla.ShardingSpec(
            sharding=(Chunked([1]), Chunked([mesh_shape[1]])),
            mesh_mapping=(ShardedAxis(0), ShardedAxis(1), Replicated(mesh_shape[0]))
        )
        # Row partitioned
        assert weight1.sharding_spec == pxla.ShardingSpec(
            sharding=(Chunked([mesh_shape[1]]), Chunked([1])),
            mesh_mapping=(ShardedAxis(0), ShardedAxis(1), Replicated(mesh_shape[0]))
        )

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
        for i in range(num_layers):
            weight = optimizer.target["params"][f"Dense_{i}"]["kernel"]
            if i % 2 == 0:
                # Column partitioned
                assert weight.sharding_spec == pxla.ShardingSpec(
                    sharding=(Chunked([1]), Chunked([mesh_shape[1]])),
                    mesh_mapping=(ShardedAxis(0), ShardedAxis(1), Replicated(mesh_shape[0]))
                )
            else:
                # Row partitioned
                assert weight.sharding_spec == pxla.ShardingSpec(
                    sharding=(Chunked([mesh_shape[1]]), Chunked([1])),
                    mesh_mapping=(ShardedAxis(0), ShardedAxis(1), Replicated(mesh_shape[0]))
                )

def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp_data_parallel'))
    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp_model_parallel'))
    suite.addTest(AutoShardingMLPTest('test_n_layer_mlp_data_parallel'))
    suite.addTest(AutoShardingMLPTest('test_n_layer_mlp_model_parallel'))

    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp_2d_mesh'))
    suite.addTest(AutoShardingMLPTest('test_n_layer_mlp_2d_mesh'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

