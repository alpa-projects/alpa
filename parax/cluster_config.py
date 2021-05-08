""" Cluster related configurations (e.g., topology) """

import numpy as np

class DeviceMesh:
    """A device mesh. Each mesh dimention has its own latency and bandwidth.
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

    def __hash__(self):
        return hash((self.flatten_devices, self.device_mesh.shape,
                self.mesh_alpha, self.mesh_beta))

    def __eq__(self, other):
        return (self.flatten_devices, self.device_mesh.shape,
                self.mesh_alpha, self.mesh_beta) ==\
               (other.flatten_devices, other.device_mesh.shape,
                other.mesh_alpha, other.mesh_beta)

