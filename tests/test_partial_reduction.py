from functools import partial
import os

import numpy as np

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, ShardedAxis

from parax import parallelize, global_config, testing


MB = 1024 ** 2

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

    hlo_module = testing.last_compiled_executable().hlo_modules()[0]
    hlo_ir = hlo_module.to_string()

    assert hlo_ir.count("all-reduce(") == 1


if __name__ == "__main__":
    global_config.set_shard_parallel_strategy('auto_sharding')

    test_all_reduce_simplification()

