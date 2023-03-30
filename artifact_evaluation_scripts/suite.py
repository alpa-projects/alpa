"""Benchmark suites for cross mesh resharding microbenchmarks."""
from collections import namedtuple
from jax.interpreters.pxla import (Chunked, NoSharding, Replicated, ShardedAxis,
                                   ShardingSpec)

BenchmarkCase = namedtuple("BenchmarkCase", [
    "src_mesh_shape", "dst_mesh_shape", "tensor_shape", "src_sharding_spec",
    "dst_sharding_spec"
])

perf_n_to_m_suite = {
    "case1":
        BenchmarkCase(
            (2, 4),
            (2, 4),
            # (1024 // 8, 1024, 512),
            (1024, 1024, 512),
            ShardingSpec([Chunked(
                [2]), NoSharding(), NoSharding()],
                         [ShardedAxis(0), Replicated(4)]),
            ShardingSpec([Chunked(
                [2]), NoSharding(), NoSharding()],
                         [ShardedAxis(0), Replicated(4)]),
        ),
    "case2":
        BenchmarkCase(
            (2, 4),
            (2, 4),
            # (1024 // 8, 1024, 512),
            (1024, 1024, 512),
            ShardingSpec(
                [NoSharding(), NoSharding(),
                 NoSharding()], [Replicated(8)]),
            ShardingSpec([Chunked(
                [2]), NoSharding(), NoSharding()],
                         [ShardedAxis(0), Replicated(4)]),
        ),
    "case3":
        BenchmarkCase(
            (2, 4),
            (2, 4),
            # (1024 // 8, 1024, 512),
            (1024, 1024, 512),
            ShardingSpec(
                [NoSharding(), Chunked([2]),
                 NoSharding()], [ShardedAxis(0), Replicated(4)]),
            ShardingSpec([Chunked(
                [2]), NoSharding(), NoSharding()],
                         [ShardedAxis(0), Replicated(4)]),
        ),
    "case4":
        BenchmarkCase(
            (2, 4),
            (2, 4),
            # (1024 // 8, 1024, 512),
            (1024, 1024, 512),
            ShardingSpec(
                [NoSharding(), Chunked([8]),
                 NoSharding()], [ShardedAxis(0)]),
            ShardingSpec([Chunked(
                [8]), NoSharding(), NoSharding()], [ShardedAxis(0)]),
        ),
    "case5":
        BenchmarkCase(
            (2, 4),
            (2, 4),
            # (1024 // 8, 1024, 512),
            (1024, 1024, 512),
            ShardingSpec([Chunked(
                [4]), NoSharding(), NoSharding()],
                         [Replicated(2), ShardedAxis(0)]),
            ShardingSpec([Chunked(
                [2]), NoSharding(), NoSharding()],
                         [ShardedAxis(0), Replicated(4)]),
        ),
    "case6":
        BenchmarkCase(
            (2, 4),
            (3, 4),
            # (1024*3//8, 1024, 170),
            (1024 * 3, 1024, 170),
            ShardingSpec([Chunked(
                [2]), NoSharding(), NoSharding()],
                         [ShardedAxis(0), Replicated(4)]),
            ShardingSpec([Chunked(
                [3]), NoSharding(), NoSharding()],
                         [ShardedAxis(0), Replicated(4)]),
        ),
    "case7":
        BenchmarkCase(
            (1, 4),
            (2, 4),
            # (1024 // 8, 1024, 512),
            (1024, 1024, 512),
            ShardingSpec([Chunked(
                [4]), NoSharding(), NoSharding()], [ShardedAxis(0)]),
            ShardingSpec(
                [NoSharding(), NoSharding(),
                 NoSharding()], [Replicated(4)]),
        ),
    "case8":
        BenchmarkCase(
            (1, 4),
            (2, 4),
            # (1024 // 8, 1024, 512),
            (1024, 1024, 512),
            ShardingSpec([Chunked(
                [4]), NoSharding(), NoSharding()], [ShardedAxis(0)]),
            ShardingSpec(
                [NoSharding(), NoSharding(),
                 NoSharding()], [Replicated(4)]),
        ),
    "case9":
        BenchmarkCase(
            (2, 4),
            (2, 4),
            # (1024 // 8, 1024, 512),
            (1024, 1024, 512),
            ShardingSpec(
                [NoSharding(), Chunked([2]),
                 NoSharding()], [ShardedAxis(0), Replicated(4)]),
            ShardingSpec(
                [NoSharding(), NoSharding(),
                 Chunked([2])], [ShardedAxis(0), Replicated(4)]),
        ),
}

resharding_n_to_m_configs = [
    {
        "resharding_mode": "send_recv",
        "resharding_loadbalance_mode": "normal",
        "use_local_allgather": False
    },
    {
        "resharding_mode": "send_recv",
        "resharding_loadbalance_mode": "normal",
        "use_local_allgather": True
    },
    {
        "resharding_mode": "broadcast",
        "resharding_loadbalance_mode": "no_loadbalance",
        "use_local_allgather": False
    },
    {
        "resharding_mode": "broadcast",
        "resharding_loadbalance_mode": "loadbalance_size",
        "use_local_allgather": False
    },
    {
        "resharding_mode": "broadcast",
        "resharding_loadbalance_mode": "loadbalance_order",
        "use_local_allgather": False
    },
]

perf_1_to_m_suite = {(n_node, gpu_per_node): BenchmarkCase(
    (1, 1),
    (n_node, gpu_per_node),
    (1 << 28,),
    ShardingSpec([NoSharding()], [Replicated(1)]),
    ShardingSpec([NoSharding()], [Replicated(n_node * gpu_per_node)]),
) for n_node, gpu_per_node in [(1, 1), (1, 2), (1, 3), (1, 4)]
                    }

resharding_1_to_m_configs = [
    {
        "resharding_mode": "send_recv",
        "resharding_loadbalance_mode": "normal",
        "use_local_allgather": False
    },
    {
        "resharding_mode": "send_recv",
        "resharding_loadbalance_mode": "normal",
        "use_local_allgather": True
    },
    {
        "resharding_mode": "broadcast",
        "resharding_loadbalance_mode": "normal",
        "use_local_allgather": False
    },
]
