from functools import partial
import pickle

import numpy as np

from jax.interpreters import pxla
from jax.interpreters.pxla import ShardingSpec, Chunked, NoSharding, Replicated, ShardedAxis


def test_order():
    a = pxla.ShardingSpec(sharding=(Chunked([2]), NoSharding()),
                          mesh_mapping=(ShardedAxis(0), Replicated(2)))

    print("--")
    print(a.indices((4, 4)).flatten()[0])
    print(a.indices((4, 4)).flatten()[1])

    b = pxla.ShardingSpec(sharding=(Chunked([2]), NoSharding()),
                          mesh_mapping=(Replicated(2), ShardedAxis(0)))

    print("--")
    print(b.indices((4, 4)).flatten()[0])
    print(b.indices((4, 4)).flatten()[1])


def test_equivalent():
    a = pxla.ShardingSpec(sharding=(Chunked([4]), Chunked([1])),
                          mesh_mapping=(ShardedAxis(0), ShardedAxis(1)))

    print("--")
    print(a.indices((4, 4)).flatten()[0])
    print(a.indices((4, 4)).flatten()[1])
    print(a.indices((4, 4)).flatten()[2])
    print(a.indices((4, 4)).flatten()[3])

    a = pxla.ShardingSpec(sharding=(Chunked([4]), NoSharding()),
                          mesh_mapping=(Replicated(1), ShardedAxis(0)))

    print("--")
    print(a.indices((4, 4)).flatten()[0])
    print(a.indices((4, 4)).flatten()[1])
    print(a.indices((4, 4)).flatten()[2])
    print(a.indices((4, 4)).flatten()[3])


def test_multiple_chunks():
    a = pxla.ShardingSpec(sharding=(Chunked([2, 2]),),
                          mesh_mapping=(ShardedAxis(1), ShardedAxis(0)))

    print(a.indices((4,)).flatten()[0])
    print(a.indices((4,)).flatten()[1])
    print(a.indices((4,)).flatten()[2])
    print(a.indices((4,)).flatten()[3])


def test_pickle():
    a = pxla.ShardingSpec(sharding=(Chunked([2, 2]),),
                          mesh_mapping=(ShardedAxis(1), ShardedAxis(0)))

    pickle.dump(a, open("tmp.pkl", "wb"))

    b = pickle.load(open("tmp.pkl", "rb"))

    assert a == b


def sharding_spec_getstate(self):
    sharding = []
    for x in self.sharding:
        if isinstance(x, pxla.NoSharding):
            sharding.append((0,))
        elif isinstance(x, pxla.Chunked):
            sharding.append((1, x.chunks))
        elif isinstance(x, pxla.Unstacked):
            sharding.append((2, x.size))
        else:
            raise ValueError(f"Invalid sharding: {x}")
    mesh_mapping = []
    for x in self.mesh_mapping:
        if isinstance(x, pxla.ShardedAxis):
            mesh_mapping.append((0, x.axis))
        elif isinstance(x, pxla.Replicated):
            mesh_mapping.append((1, x.replicas))
        else:
            raise ValueError(f"Invalid sharding: {x}")
    return (sharding, mesh_mapping)


def sharding_spec_setstate(self, state_tuple):
    sharding_encoding, mesh_mapping_encoding = state_tuple

    sharding = []
    for x in sharding_encoding:
        if x[0] == 0:
            sharding.append(pxla.NoSharding())
        elif x[0] == 1:
            sharding.append(pxla.Chunked(x[1]))
        elif x[0] == 2:
            sharding.append(pxla.Unstacked(x[1]))
        else:
            raise ValueError(f"Invalid sharding: {x}")

    mesh_mapping = []
    for x in mesh_mapping_encoding:
        if x[0] == 0:
            mesh_mapping.append(pxla.ShardedAxis(x[1]))
        elif x[0] == 1:
            mesh_mapping.append(pxla.Replicated(x[1]))
        else:
            raise ValueError(f"Invalid sharding: {x}")

    self.__init__(
        sharding=sharding,
        mesh_mapping=mesh_mapping,
    )


setattr(pxla.ShardingSpec, "__getstate__", sharding_spec_getstate)
setattr(pxla.ShardingSpec, "__setstate__", sharding_spec_setstate)

if __name__ == "__main__":
    #test_order()
    #test_equivalent()
    #test_multiple_chunks()
    test_pickle()
