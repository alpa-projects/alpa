"""Cluster Environment"""
from hlo import ShardingSpec, ShardingSpecType
from common import compute_bytes


class ClusterEnvironment:
    def __init__(self, num_devices, memory_per_device):
        self.num_devices = num_devices
        self.memory_per_device = memory_per_device
        self.alpha = 1
        self.beta = 1
        self.all_reduce_penalty = 0
        self.all_gather_penalty = 0
        self.reduce_scatter_penalty = 0
        self.partial_reduction_penalty = 0

    def all_reduce_cost(self, num_bytes):
        return (self.alpha +
                self.beta * 2 * (self.num_devices - 1) / self.num_devices * num_bytes +
                0.1) + self.all_reduce_penalty

    def all_gather_cost(self, num_bytes):
        return (self.alpha +
                self.beta * (self.num_devices - 1) / self.num_devices * num_bytes +
                0.01) + self.all_gather_penalty

    def reduce_scatter_cost(self, num_bytes):
        return (self.alpha +
                self.beta * (self.num_devices - 1) / self.num_devices * num_bytes +
                0.001) + self.reduce_scatter_penalty

    def resharding_cost(self, shape, src_spec, dst_spec):
        if src_spec == dst_spec:
            return 0

        if src_spec.type == ShardingSpecType.REPLICATED:
            if dst_spec.type == ShardingSpecType.PARTIAL_REDUCTION:
                # An elementwise divide will occur here,
                # so we should add a penanlty.
                return self.partial_reduction_penalty
            else:
                return 0

        if src_spec.type == ShardingSpecType.PARTIAL_REDUCTION:
            ret = self.all_reduce_cost(compute_bytes(shape))
            return ret + self.partial_reduction_penalty

        return self.all_gather_cost(compute_bytes(shape))

