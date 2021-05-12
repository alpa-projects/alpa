""" Cluster related configurations (e.g., topology) """

import numpy as np

from parax.util import get_dim_last_value


class LogicalDeviceMesh:
    """A multi-dimensional device mesh topology.
    Each mesh dimension has its own latency and bandwidth.
    We use alpha-beta model to model the communication cost.
    """
    def __init__(self, device_id_mesh, mesh_alpha=None, mesh_beta=None, is_multi_host=False):
        self.id_mesh = device_id_mesh
        self.flatten_ids = tuple(self.id_mesh.flatten())
        self.is_multi_host = False

        # coefficient for alpha-beta communication model
        if mesh_alpha is None:
            mesh_alpha = [1] * len(self.id_mesh.shape)
        if mesh_beta is None:
            mesh_beta = [1] * len(self.id_mesh.shape)
        self.mesh_alpha = tuple(mesh_alpha)
        self.mesh_beta = tuple(mesh_beta)

    def all_gather_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
                0.1)

    def all_reduce_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * 2 * (num_devices - 1) / num_devices * num_bytes +
                0.01)

    def reduce_scatter_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
                0.001)

    def get_tensor_dim_to_mesh_dim(self, shape,
                                   tile_assignment_dimensions, tile_assignment_devices):
        tile_assignment = np.array(tile_assignment_devices).reshape(tile_assignment_dimensions)

        tensor_dim_vals = tuple(get_dim_last_value(tile_assignment, i)
            for i in range(len(shape)))

        mesh_dim_vals = tuple(get_dim_last_value(self.id_mesh, j)
            for j in range(len(self.id_mesh.shape)))

        ret = [-1] * len(shape)
        for i in range(len(shape)):
            if tile_assignment_dimensions[i] != 1:
                found = False
                for j in range(len(self.id_mesh.shape)):
                    if tensor_dim_vals[i] == mesh_dim_vals[j]:
                        ret[i] = j
                        found = True
                assert found

        return ret

    def __hash__(self):
        return hash((self.flatten_ids, self.id_mesh.shape,
                     self.mesh_alpha, self.mesh_beta))

    def __eq__(self, other):
        return (self.flatten_ids, self.id_mesh.shape,
                self.mesh_alpha, self.mesh_beta) ==\
               (other.flatten_ids, other.id_mesh.shape,
                other.mesh_alpha, other.mesh_beta)

