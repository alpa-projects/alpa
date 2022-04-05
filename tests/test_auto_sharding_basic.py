"""Test auto sharding with simple computational graphs."""

import unittest

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, ShardedAxis, NoSharding, Replicated
from flax import linen as nn
from flax import optim

from alpa import parallelize, set_parallelize_options, testing

from test_auto_sharding_mlp import assert_close

MB = 1024**2


class AutoShardingBasicTest(unittest.TestCase):

    def setUp(self):
        assert len(jax.local_devices()) >= 4
        set_parallelize_options(jax.local_devices()[:4])

    def test_donate_buffer(self):

        @parallelize(donate_argnums=(0,))
        def add_one(x):
            x = x + 1
            return x

        a = jnp.ones((128, 128))
        b = add_one(a)

        # Assert b is sharded
        assert (b.sharding_spec == pxla.ShardingSpec(
            sharding=(NoSharding(), Chunked([4])),
            mesh_mapping=(ShardedAxis(0),)) or b.sharding_spec
                == pxla.ShardingSpec(sharding=(Chunked([4]), NoSharding()),
                                     mesh_mapping=(ShardedAxis(0),)))

    def test_dot_reshape_transpose(self):
        set_parallelize_options(memory_budget_per_device=1 * MB)
        dim_0 = 64
        dim_1 = 1024

        def func(a, b):
            a = jnp.transpose(a, [0, 2, 1])
            a = jnp.reshape(a, (dim_0, dim_1))
            b = jnp.reshape(b, (dim_1, dim_0))
            out = a @ b
            out = -out
            return out

        para_func = parallelize(func)

        a = jnp.ones((dim_0, dim_1 // 4, 4))
        b = jnp.ones((dim_1, dim_0 // 4, 4))

        # Check correctness
        expected = func(a, b)
        actual = para_func(a, b)
        testing.assert_allclose(expected, actual)

    def test_one_by_one_mesh(self):
        set_parallelize_options(devices=jax.local_devices()[0:1])

        @parallelize
        def add_one(x):
            x = x + 1
            return x

        a = jnp.ones((128, 128))
        b = add_one(a)

        testing.assert_allclose(b, a + 1)

    def test_dropout(self):

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x, deterministic):
                x = nn.Dense(16, use_bias=False)(x)
                x = nn.Dropout(0.1, deterministic=deterministic)(x)
                x = nn.Dense(16, use_bias=False)(x)
                return x

        x = jnp.ones((32, 32, 16))
        y = jnp.ones((32, 32, 16))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, True)
        optimizer = optim.GradientDescent(1e-2).create(params)

        @parallelize
        def func(optimizer, x, y, rngs):

            def loss_func(params):
                out = model.apply(params, x, False, rngs=rngs)
                return jnp.mean((out - y)**2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        # Check sharding strategy (data-parallel)
        executable = func.get_executable(optimizer, x, y, {"dropout": rngkey})
        assert executable.auto_sharding_objective < 1e6

        hlo_ir = executable.get_hlo_text()
        assert "u64[1024]{0} iota()" in hlo_ir  # 1024 = 32 * 32 * 16 / 4 / 4
        assert hlo_ir.count("channel_id") == 1
        assert hlo_ir.count("all-reduce(") == 1

    def test_gather(self):

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(32, use_bias=False)(x)
                idx = jnp.arange(16)
                x = x[:,idx]
                x = nn.Dense(16, use_bias=False)(x)
                return x

        x = jnp.ones((256, 32))
        y = jnp.ones((256, 16))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        @parallelize
        def func(optimizer, x, y):

            def loss_func(params):
                out = model.apply(params, x)
                return jnp.mean((out - y)**2)

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        executable = func.get_executable(optimizer, x, y)
        assert executable.auto_sharding_objective < 1e6

        hlo_ir = executable.get_hlo_text()
        assert "gather(f32[64,32]" in hlo_ir
        assert "scatter(f32[64,32]" in hlo_ir
        assert hlo_ir.count("all-reduce(") == 1

    def test_reshape_uneven_partition(self):
        # TODO(lmzheng): Support the uneven partition of reshape.
        # But this seems too complicated.

        set_parallelize_options(devices=jax.local_devices()[0:4])

        @parallelize
        def split(a):
            b = a.reshape((8, 18))
            #b = a.reshape((9, 16))
            return b

        a = jnp.ones(144)
        split(a)

        executable = split.get_executable(a)
        assert_close(executable.auto_sharding_objective, 0)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingBasicTest("test_donate_buffer"))
    suite.addTest(AutoShardingBasicTest("test_dot_reshape_transpose"))
    suite.addTest(AutoShardingBasicTest("test_one_by_one_mesh"))
    suite.addTest(AutoShardingBasicTest("test_gather"))
    suite.addTest(AutoShardingBasicTest("test_dropout"))
    suite.addTest(AutoShardingBasicTest("test_reshape_uneven_partition"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
