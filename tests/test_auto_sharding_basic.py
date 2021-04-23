from functools import partial
import os

import numpy as np

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, ShardedAxis

from parax import parallelize, global_config, testing


MB = 1024 ** 2

def assert_close(x, y):
    assert abs(x / y - 1) < 0.001, f"{x} vs. {y}"


def test_donate_buffer():
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    @parallelize(donate_argnums=(0,),
                 memory_budget_per_device=3 * MB,
                 devices=devices)
    def add_one(x):
        x = x + 1
        return x

    a = jnp.ones((1024, 1024))
    b = add_one(a)

    hlo_module = testing.last_compiled_executable.hlo_modules()[0]
    hlo_ir = hlo_module.to_string()

    # assert a and b are split over the second dimension
    assert b.sharding_spec == pxla.ShardingSpec(
        sharding=(Chunked([1]), Chunked([4])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1))) or\
           b.sharding_spec == pxla.ShardingSpec(
        sharding=(Chunked([4]), Chunked([1])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1)))


def test_dot_reshape_transpose():
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    dim_0 = 128
    dim_1 = 2048

    def func(a, b):
        a = jnp.reshape(a, (dim_0, dim_1))
        b = jnp.reshape(b, (dim_1, dim_0))
        out = a @ b
        out = -out
        return out

    para_func = parallelize(memory_budget_per_device=2 * MB, devices=devices)(func)

    a = jnp.ones((dim_0, 4, dim_1 // 4))
    b = jnp.ones((dim_1, dim_0 // 4, 4))

    expected = func(a, b)
    actual = para_func(a, b)

    np.testing.assert_allclose(expected, actual)


def test_all_reduce_simplification():
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    dim_0 = 128
    dim_1 = 2048

    def func(a, b, c, d, e, f):
        h1 = a @ b
        h2 = c @ d
        h3 = e @ f
        out = h1 + h2 + h3
        out = jnp.exp(out)
        return out

    para_func = parallelize(memory_budget_per_device=2 * MB, devices=devices)(func)

    a = jnp.ones((dim_0, dim_1))
    b = jnp.ones((dim_1, dim_0))
    c = jnp.ones((dim_0, dim_1))
    d = jnp.ones((dim_1, dim_0))
    e = jnp.ones((dim_0, dim_1))
    f = jnp.ones((dim_1, dim_0))

    expected = func(a, b, c, d, e, f)
    actual = para_func(a, b, c, d, e, f)

    np.testing.assert_allclose(expected, actual)

    all_reduce_cost = 2 * (len(devices) - 1) / len(devices) * dim_0 * dim_0 * 4
    assert_close(testing.last_compiled_auto_sharding_objective,
                 all_reduce_cost)


def test_all_reduce_simplification_out_reuse():
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

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

    para_func = parallelize(memory_budget_per_device=2 * MB, devices=devices)(func)

    a = jnp.ones((dim_0, dim_1))
    b = jnp.ones((dim_1, dim_0))
    c = jnp.ones((dim_0, dim_1))
    d = jnp.ones((dim_1, dim_0))
    e = jnp.ones((dim_0, dim_1))
    f = jnp.ones((dim_1, dim_0))
    g = jnp.ones((dim_0 // 4, 4, dim_0))

    expected = func(a, b, c, d, e, f, g)
    actual = para_func(a, b, c, d, e, f, g)

    np.testing.assert_allclose(expected, actual)

    all_reduce_cost = 2 * (len(devices) - 1) / len(devices) * dim_0 * dim_0 * 4
    assert_close(testing.last_compiled_auto_sharding_objective,
                 all_reduce_cost)


if __name__ == "__main__":
    global_config.set_shard_parallel_strategy('auto_sharding')

    test_donate_buffer()
    test_dot_reshape_transpose()
    test_all_reduce_simplification()
    test_all_reduce_simplification_out_reuse()

