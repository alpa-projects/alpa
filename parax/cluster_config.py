""" Cluster related configurations (e.g., topology) """

import numpy as np

class DeviceMesh:
    """A multi-dimensional device mesh topology.
    Each mesh dimension has its own latency and bandwidth.
    We use alpha-beta model to model the communication cost.
    """
    def __init__(self, device_mesh, mesh_alpha=None, mesh_beta=None):
        self.device_mesh = np.array(device_mesh)
        self.flatten_devices = tuple(self.device_mesh.flatten())

        if mesh_alpha is None:
            mesh_alpha = [1] * len(self.device_mesh.shape)
        if mesh_beta is None:
            mesh_beta = [1] * len(self.device_mesh.shape)
        self.mesh_alpha = tuple(mesh_alpha)
        self.mesh_beta = tuple(mesh_beta)

    def all_gather_cost(self, num_bytes, mesh_dim):
        num_devices = self.device_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
                0.1)

    def all_reduce_cost(self, num_bytes, mesh_dim):
        num_devices = self.device_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * 2 * (num_devices - 1) / num_devices * num_bytes +
                0.01)

    def reduce_scatter_cost(self, num_bytes, mesh_dim):
        num_devices = self.device_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] +
                self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
                0.001)

    def __hash__(self):
        return hash((self.flatten_devices, self.device_mesh.shape,
                self.mesh_alpha, self.mesh_beta))

    def __eq__(self, other):
        return (self.flatten_devices, self.device_mesh.shape,
                self.mesh_alpha, self.mesh_beta) ==\
               (other.flatten_devices, other.device_mesh.shape,
                other.mesh_alpha, other.mesh_beta)

