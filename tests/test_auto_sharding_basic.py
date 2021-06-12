"""Test auto sharding with simple computational graphs."""

from functools import partial
import os
import unittest

import numpy as np

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, ShardedAxis, NoSharding, Replicated
from flax import linen as nn
from flax import optim

from parax import parallelize, global_config, testing

from test_auto_sharding_mlp import assert_close, all_reduce_cost

MB = 1024 ** 2

class AutoShardingBasicTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = tuple(jax.local_devices()[:4])
        global_config.shard_parallel_strategy = "auto_sharding"


    def test_one_by_one_mesh(self):
        @parallelize(donate_argnums=(0,),
                     devices=self.devices[0:1])
        def add_one(x):
            x = x + 1
            return x

        a = jnp.ones((128, 128))
        b = add_one(a)

        np.testing.assert_allclose(b, a + 1)


    def test_donate_buffer(self):
        @parallelize(donate_argnums=(0,),
                     devices=self.devices)
        def add_one(x):
            x = x + 1
            return x

        a = jnp.ones((128, 128))
        b = add_one(a)

        # Check sharding strategy
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()

        # Assert b is sharded
        assert b.sharding_spec == pxla.ShardingSpec(
            sharding=(NoSharding(), Chunked([4])),
            mesh_mapping=(Replicated(1), ShardedAxis(0))) or\
               b.sharding_spec == pxla.ShardingSpec(
            sharding=(Chunked([4]), NoSharding()),
            mesh_mapping=(Replicated(1), ShardedAxis(0)))


    def test_dropout(self):
        class Model(nn.Module):
            @nn.compact
            def __call__(self, x, deterministic):
                x = nn.Dense(128, use_bias=False)(x)
                x = nn.Dropout(0.1, deterministic=deterministic)(x)
                return x

        x = jnp.ones((256, 256, 128))
        y = jnp.ones((256, 256, 128))

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, True)
        optimizer = optim.GradientDescent(1e-2).create(params)

        @parallelize
        def func(optimizer, x, y, rngs):
            def loss_func(params):
                out = model.apply(params, x, False, rngs=rngs)
                return jnp.mean((out - y) ** 2)
            grad = jax.grad(loss_func)(optimizer.target)
            return grad

        expected = func(optimizer, x, y, {"dropout": rngkey})

        # Check sharding strategy
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()


    def test_dot_reshape_transpose(self):
        dim_0 = 64
        dim_1 = 1024

        def func(a, b):
            a = jnp.reshape(a, (dim_0, dim_1))
            b = jnp.reshape(b, (dim_1, dim_0))
            out = a @ b
            out = -out
            return out

        para_func = parallelize(memory_budget_per_device=1 * MB, devices=self.devices)(func)

        a = jnp.ones((dim_0, 4, dim_1 // 4))
        b = jnp.ones((dim_1, dim_0 // 4, 4))

        # Check correctness
        expected = func(a, b)
        actual = para_func(a, b)
        np.testing.assert_allclose(expected, actual)

    def test_all_reduce_simplification(self):
        # This test is deprecated.
        # This requires partial_reduction, which is not in our current plan
        return

        dim_0 = 128
        dim_1 = 2048

        def func(a, b, c, d, e, f):
            h1 = a @ b
            h2 = c @ d
            h3 = e @ f
            out = h1 + h2 + h3
            out = jnp.exp(out)
            return out

        para_func = parallelize(memory_budget_per_device=2 * MB, devices=self.devices)(func)

        a = jnp.ones((dim_0, dim_1))
        b = jnp.ones((dim_1, dim_0))
        c = jnp.ones((dim_0, dim_1))
        d = jnp.ones((dim_1, dim_0))
        e = jnp.ones((dim_0, dim_1))
        f = jnp.ones((dim_1, dim_0))

        # Check correctness
        expected = func(a, b, c, d, e, f)
        actual = para_func(a, b, c, d, e, f)
        np.testing.assert_allclose(expected, actual)

        # Check sharding strategy
        expected = all_reduce_cost(len(self.devices), dim_0 * dim_0 * 4)
        assert_close(testing.last_compiled_auto_sharding_objective, expected)

    def test_all_reduce_simplification_out_reuse(self):
        # This test is deprecated.
        # This requires partial_reduction, which is not in our current plan
        return

        dim_0 = 128
        dim_1 = 2048

        def func(a, b, c, d, e, f, g):
            h1 = a @ b
            h2 = c @ d
            h3 = e @ f
            h1 = jnp.reshape(h1, [dim_0 // 4, 4, dim_0])
            h2 = jnp.reshape(h2, [dim_0 // 4, 4, dim_0])
            h3 = jnp.reshape(h3, [dim_0 // 4, 4, dim_0])
            out = jnp.negative(g)
            out = out + h1
            out = out + h2
            out = out + h3
            out = jnp.negative(out)
            return out

        para_func = parallelize(memory_budget_per_device=2 * MB, devices=self.devices)(func)

        a = jnp.ones((dim_0, dim_1))
        b = jnp.ones((dim_1, dim_0))
        c = jnp.ones((dim_0, dim_1))
        d = jnp.ones((dim_1, dim_0))
        e = jnp.ones((dim_0, dim_1))
        f = jnp.ones((dim_1, dim_0))
        g = jnp.ones((dim_0 // 4, 4, dim_0))

        # Check correctness
        expected = func(a, b, c, d, e, f, g)
        actual = para_func(a, b, c, d, e, f, g)
        np.testing.assert_allclose(expected, actual)

        # Check sharding strategy
        expected = all_reduce_cost(len(self.devices), dim_0 * dim_0 * 4)
        assert_close(testing.last_compiled_auto_sharding_objective, expected)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingBasicTest('test_one_by_one_mesh'))
    suite.addTest(AutoShardingBasicTest('test_donate_buffer'))
    suite.addTest(AutoShardingBasicTest('test_dropout'))
    suite.addTest(AutoShardingBasicTest('test_dot_reshape_transpose'))
    suite.addTest(AutoShardingBasicTest('test_all_reduce_simplification'))
    suite.addTest(AutoShardingBasicTest('test_all_reduce_simplification_out_reuse'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

