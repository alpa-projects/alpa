"""Cluster Environment"""
import numpy as np

from hlo import ShardingSpec, ShardingSpecType
from common import compute_bytes, get_dim_last_value


class ClusterEnvironment:
    def __init__(self, device_mesh, mesh_alpha, mesh_beta, memory_per_device, solver_option=None):
        self.device_mesh = np.array(device_mesh)
        self.mesh_alpha = mesh_alpha
        self.mesh_beta = mesh_beta
        assert len(self.mesh_alpha) == len(self.device_mesh.shape)
        assert len(self.mesh_beta) == len(self.device_mesh.shape)
        self.memory_per_device = memory_per_device
        self.all_gather_penalty = 0
        self.all_reduce_penalty = 0
        self.reduce_scatter_penalty = 0
        self.partial_reduction_penalty = 10
        self.num_devices = np.prod(self.device_mesh.shape)

        self.force_all_gather_cost = None
        self.force_all_reduce_cost = None
        self.force_reduce_scatter_cost = None

        if solver_option:
            self.force_all_gather_cost = solver_option.force_all_gather_cost
            self.force_all_reduce_cost = solver_option.force_all_reduce_cost
            self.force_reduce_scatter_cost = solver_option.force_reduce_scatter_cost

    def all_gather_cost(self, num_bytes, mesh_dim=0):
        if self.force_all_gather_cost:
            return self.force_all_gather_cost

        num_devices = self.device_mesh.shape[mesh_dim]
        return (int(self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes) +
                0.1) + self.all_gather_penalty

    def all_reduce_cost(self, num_bytes, mesh_dim=0):
        if self.force_all_reduce_cost:
            return self.force_all_reduce_cost

        num_devices = self.device_mesh.shape[mesh_dim]
        return (int(self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * 2 * (num_devices - 1) / num_devices * num_bytes) +
                0.01) + self.all_reduce_penalty

    def reduce_scatter_cost(self, num_bytes, mesh_dim=0):
        if self.force_reduce_scatter_cost:
            return self.force_reduce_scatter_cost

        num_devices = self.device_mesh.shape[mesh_dim]
        return (int(self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes) +
                0.001)

    def all_to_all_cost(self, num_bytes, mesh_dim=0):
        num_devices = self.device_mesh.shape[mesh_dim]
        penalty_factor = 1.5;
        return (int(self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices /\
                    num_devices * num_bytes * penalty_factor) +
                0.001);

    def get_tensor_dim_to_mesh_dim(self, shape, spec):
        """Map the tensor dimention to mesh dimension, -1 means replicated"""
        if spec.type == ShardingSpecType.REPLICATED:
            return [-1] * len(shape)

        tile_assignment = np.array(spec.tile_assignment_devices).\
            reshape(spec.tile_assignment_dimensions)

        tensor_dim_vals = tuple(get_dim_last_value(tile_assignment, i)
            for i in range(len(shape)))

        mesh_dim_vals = tuple(get_dim_last_value(self.device_mesh, j)
            for j in range(len(self.device_mesh.shape)))

        ret = [-1] * len(shape)
        for i in range(len(shape)):
            if spec.tile_assignment_dimensions[i] != 1:
                found = False
                for j in range(len(self.device_mesh.shape)):
                    if tensor_dim_vals[i] == mesh_dim_vals[j]:
                        ret[i] = j
                        found = True
                assert found

        return ret

    def resharding_cost(self, shape, src_spec, dst_spec):
        if src_spec == dst_spec:
            return 0

        src_tensor_dim_to_mesh_dim = self.get_tensor_dim_to_mesh_dim(shape, src_spec)
        dst_tensor_dim_to_mesh_dim = self.get_tensor_dim_to_mesh_dim(shape, dst_spec)

        cost = 0
        for i in range(len(shape)):
            src_mesh_dim = src_tensor_dim_to_mesh_dim[i]
            if src_mesh_dim == -1:
                continue
            if src_mesh_dim == dst_tensor_dim_to_mesh_dim[i]:
                continue
            cost += self.all_gather_cost(compute_bytes(shape), src_mesh_dim)

        return cost

