import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import (ShardingSpec,
    NoSharding, Replicated, Chunked, ShardedAxis)
import numpy as np
import ray

import alpa

def benchmark(physical_mesh, shape, sharding_spec):
    avals = []
    shard_indices = []
    sharding_specs = []
    donated_invars = []
    args = []

    number = 2

    for i in range(number):
        array = jnp.ones(shape, jnp.float32)
        indices = sharding_spec.indices(array.shape)

        avals.append(jax.ShapedArray(array.shape, array.dtype))
        sharding_specs.append(sharding_spec)
        shard_indices.append(indices.flatten())
        donated_invars.append(True)
        args.append(array)

    print(sharding_spec)
    buffers = physical_mesh.shard_args_to_bufs(shard_indices, donated_invars, args)

    return buffers


if __name__ == "__main__":
    ray.init(address="auto")

    cluster = alpa.DeviceCluster()
    physical_mesh = cluster.get_physical_mesh()

    shape = (8192, 8192)

    sharding_specs = [
        ShardingSpec(
            sharding=[NoSharding(), NoSharding(),],
            mesh_mapping=[Replicated(8),]),
        ShardingSpec(
            sharding=[Chunked([8]), NoSharding(),],
            mesh_mapping=[ShardedAxis(0),]),
        ShardingSpec(
            sharding=[NoSharding(), Chunked([8])],
            mesh_mapping=[ShardedAxis(0),]),
        ShardingSpec(
            sharding=[Chunked([2]), Chunked([4])],
            mesh_mapping=[ShardedAxis(0), ShardedAxis(1)]),
    ]

    for spec in sharding_specs:
        benchmark(physical_mesh, shape, spec)

