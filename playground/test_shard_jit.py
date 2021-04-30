from functools import partial

import numpy as np

from jax.experimental import (sharded_jit, with_sharding_constraint,
    PartitionSpec as P)
from jax.interpreters import pxla


def test_basic():
    @partial(sharded_jit, in_parts=(P(2,1), P(2,1)), out_parts=None)
    def f(x, y):
        return x + y

    shape = (4, 2)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    out = f(x, x + 1)

    print(out.sharding_spec)


if __name__ == "__main__":
    test_basic()

