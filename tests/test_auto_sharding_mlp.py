import numpy as np

import unittest

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis
from flax import linen as nn
from flax import optim

from parax import parallelize, global_config, testing

from test_auto_sharding_basic import assert_close, all_reduce_cost

MB = 1024 ** 2

class AutoShardingMLPTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = tuple(jax.local_devices()[:4])
        global_config.shard_parallel_strategy = "auto_sharding"

    def test_2_layer_mlp(self):
        global_config.auto_sharding_solver_strategy = 'normal'

        class Model(nn.Module):
            hidden_dim: int
            output_dim: int

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=self.hidden_dim)(x)
                x = nn.relu(x)
                x = nn.Dense(features=self.output_dim)(x)
                return x

        @parallelize(memory_budget_per_device=50 * (1 << 20),
                     devices=self.devices)
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y']) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        batch_size = 128
        hidden_dim = 2048
        input_dim = output_dim = hidden_dim

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = Model(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compile
        optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

        # Check sharding strategy
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        # The function should contain only one communication primitive,
        # which is an all-reduce
        assert hlo_ir.count("channel_id") == 1, hlo_ir.count("channel_id")
        assert hlo_ir.count("all-reduce(") == 1

        expected = all_reduce_cost(len(self.devices), batch_size * hidden_dim * 4)
        assert_close(testing.last_compiled_auto_sharding_objective, expected)

        weight0 = optimizer.target["params"]["Dense_0"]["kernel"]
        weight1 = optimizer.target["params"]["Dense_1"]["kernel"]
        assert isinstance(weight0, pxla.ShardedDeviceArray)
        assert isinstance(weight1, pxla.ShardedDeviceArray)
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

    def test_n_layer_mlp(self):
        global_config.auto_sharding_solver_strategy = 'normal'

        assert len(jax.devices()) >= 4
        devices = tuple(jax.devices()[:4])

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

        @parallelize(memory_budget_per_device=100 * (1 << 20),
                     devices=devices)
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y']) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        batch_size = 128
        hidden_dim = 2048
        num_layers = 6
        input_dim = output_dim = hidden_dim

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

        # Check sharding strategy
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        # The function should contain 5 all-reduce
        assert hlo_ir.count("channel_id") == 5
        assert hlo_ir.count("all-reduce(") == 5

        expected = 5 * all_reduce_cost(len(self.devices), batch_size * hidden_dim * 4)
        assert_close(testing.last_compiled_auto_sharding_objective, expected)

        for i in range(num_layers):
            weight = optimizer.target["params"][f"Dense_{i}"]["kernel"]
            assert isinstance(weight, pxla.ShardedDeviceArray)
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

    def test_2_layer_mlp_force_data_parallel(self):
        global_config.auto_sharding_solver_strategy = 'force_data_parallel'

        class Model(nn.Module):
            hidden_dim: int
            output_dim: int

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=self.hidden_dim)(x)
                x = nn.relu(x)
                x = nn.Dense(features=self.output_dim)(x)
                return x

        @parallelize(memory_budget_per_device=1000 * (1 << 20),
                     devices=self.devices)
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch['x'])
                return jnp.mean((out - batch['y']) ** 2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        batch_size = 128
        hidden_dim = 2048
        input_dim = output_dim = hidden_dim

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = Model(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compiler
        optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

        # Check sharding strategy
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()
        # The function should contain only one communication primitive,
        # which is an all-reduce
        assert hlo_ir.count("channel_id") == 2, hlo_ir.count("channel_id")

        forced_all_reduce_cost = 1000
        num_weight_tensors = len(jax.tree_util.tree_leaves(params))
        expected = forced_all_reduce_cost * num_weight_tensors
        assert_close(testing.last_compiled_auto_sharding_objective, expected)


        weight0 = optimizer.target["params"]["Dense_0"]["kernel"]
        weight1 = optimizer.target["params"]["Dense_1"]["kernel"]
        assert isinstance(weight0, pxla.ShardedDeviceArray)
        assert isinstance(weight1, pxla.ShardedDeviceArray)
        assert weight0.sharding_spec == pxla.ShardingSpec(
            sharding=(NoSharding(), NoSharding()),
            mesh_mapping=(Replicated(4),),
        )
        assert weight1.sharding_spec == pxla.ShardingSpec(
            sharding=(NoSharding(), NoSharding()),
            mesh_mapping=(Replicated(4),),
        )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp'))
    suite.addTest(AutoShardingMLPTest('test_n_layer_mlp'))
    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp_force_data_parallel'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

