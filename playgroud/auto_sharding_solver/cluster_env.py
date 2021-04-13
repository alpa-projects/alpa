"""Cluster Environment"""
from hlo import ShardingSpec, ShardingSpecType
from common import compute_bytes


class ClusterEnvironment:
    def __init__(self, num_devices, memory_per_device):
        self.num_devices = num_devices
        self.memory_per_device = memory_per_device
        self.alpha = 1
        self.beta = 1

    def all_reduce_cost(self, num_bytes):
        return self.alpha + \
               self.beta * 2 * (self.num_devices - 1) / self.num_devices * num_bytes + \
               0.01

    def all_gather_cost(self, num_bytes):
        return self.alpha + \
               self.beta * (self.num_devices - 1) / self.num_devices * num_bytes + \
               0.001

    def resharding_cost(self, shape, src_spec, dst_spec):
        if src_spec == dst_spec:
            return 0
        if src_spec.type == ShardingSpecType.REPLICATED:
            return 0

        return self.all_gather_cost(compute_bytes(shape))

