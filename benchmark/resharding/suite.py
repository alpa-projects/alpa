"""Benchmark suites for cross mesh resharding microbenchmarks."""
from collections import namedtuple
from jax.interpreters.pxla import (Chunked, NoSharding, Replicated, ShardedAxis,
                                   ShardingSpec)

BenchmarkCase = namedtuple("BenchmarkCase", [
    "src_mesh_shape", "dst_mesh_shape", "tensor_shape", 
    "src_sharding_spec", "dst_sharding_spec"
])

perf_loadbalance_suite = {
    "case1": BenchmarkCase(
                (2, 4),
                (2, 4),
                (1024, 1024, 512),
                ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
                ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
            ),
    "case2": BenchmarkCase(
                (2, 4),
                (2, 4),
                (1024, 1024, 512),
                ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(8)]), 
                ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
            ),
    "case3": BenchmarkCase(
                (2, 4),
                (2, 4),
                (1024, 1024, 512),
                ShardingSpec([NoSharding(), Chunked([2]), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
                ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
            ),
    "case4": BenchmarkCase(
                (2, 4),
                (2, 4),
                (1024, 1024, 512),
                ShardingSpec([NoSharding(), Chunked([8]), NoSharding()], [ShardedAxis(0)]), 
                ShardingSpec([Chunked([8]), NoSharding(), NoSharding()], [ShardedAxis(0)]), 
            ),
    "case5": BenchmarkCase(
                (2, 4),
                (2, 4),
                (1024, 1024, 512),
                ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [Replicated(2), ShardedAxis(0)]), 
                ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
            ),
    "case6": BenchmarkCase(
                (2, 4),
                (3, 4),
                (1024*3, 1024, 170),
                ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
                ShardingSpec([Chunked([3]), NoSharding(), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
            ),
    "case7": BenchmarkCase(
                (1, 4),
                (2, 4),
                (1024, 1024, 512),
                ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)]), 
                ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)]), 
            ),
    "case8": BenchmarkCase(
                (1, 4),
                (2, 4),
                (1024, 1024, 512),
                ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)]), 
                ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)]), 
            ),
    "case9": BenchmarkCase(
                (2, 4),
                (2, 4),
                (1024, 1024, 512),
                ShardingSpec([NoSharding(), Chunked([2]), NoSharding()], [ShardedAxis(0), Replicated(4)]), 
                ShardingSpec([NoSharding(), NoSharding(), Chunked([2])], [ShardedAxis(0), Replicated(4)]), 
            ),
}

perf_broadcast_suite = {
    (i, j): BenchmarkCase(
                (1, 1),
                (i, j),
                (1024, 1024, 512),
                ShardingSpec([NoSharding()], [Replicated(1)]), 
                ShardingSpec([NoSharding()], [Replicated(i*j)]), 
            )
    for i in range(1,5) for j in range(1, 5)
}