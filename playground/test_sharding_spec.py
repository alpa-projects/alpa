from functools import partial

import numpy as np

from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, NoSharding, Replicated, ShardedAxis


def test_order():
    a = pxla.ShardingSpec(
        sharding=(Chunked([2]), NoSharding()),
        mesh_mapping=(ShardedAxis(0), Replicated(2))
    )

    print("--")
    print(a.indices((4, 4)).flatten()[0])
    print(a.indices((4, 4)).flatten()[1])

    b = pxla.ShardingSpec(
        sharding=(Chunked([2]), NoSharding()),
        mesh_mapping=(Replicated(2), ShardedAxis(0))
    )

    print("--")
    print(b.indices((4, 4)).flatten()[0])
    print(b.indices((4, 4)).flatten()[1])


def test_equivalent():
    a = pxla.ShardingSpec(
        sharding=(Chunked([4]), Chunked([1])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1))
    )

    print("--")
    print(a.indices((4, 4)).flatten()[0])
    print(a.indices((4, 4)).flatten()[1])
    print(a.indices((4, 4)).flatten()[2])
    print(a.indices((4, 4)).flatten()[3])

    a = pxla.ShardingSpec(
        sharding=(Chunked([4]), NoSharding()),
        mesh_mapping=(Replicated(1), ShardedAxis(0))
    )

    print("--")
    print(a.indices((4, 4)).flatten()[0])
    print(a.indices((4, 4)).flatten()[1])
    print(a.indices((4, 4)).flatten()[2])
    print(a.indices((4, 4)).flatten()[3])


if __name__ == "__main__":
    #test_order()
    test_equivalent()

