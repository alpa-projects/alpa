""" Cluster related configurations (e.g., topology) """

import numpy as np
import ray
from jax.lib import xla_client

from parax.util import get_dim_last_value


class LogicalDeviceMesh:
    """A multi-dimensional device mesh topology.
    Each mesh dimension has its own latency and bandwidth.
    We use alpha-beta model to model the communication cost.
    """
    def __init__(self, physical_mesh, id_mesh, mesh_alpha=None, mesh_beta=None):
        self.physical_mesh = physical_mesh
        self.id_mesh = np.array(id_mesh)
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


class SingleHostDeviceMesh:
    """A physical device mesh to present devices on a single node"""
    def __init__(self, devices):
        self.devices = devices

    def get_logical_mesh(self, mesh_shape, mesh_alpha, mesh_beta):
        device_ids = np.array([d.id for d in self.devices])
        device_ids = device_ids.reshape(mesh_shape)
        return LogicalDeviceMesh(self, device_ids, mesh_alpha, mesh_beta)


class MultiHostDeviceMesh:
    """A physical device mesh to present a device mesh on multipe nodes"""
    def __init__(self, host_ids, host_info, num_devices_per_host, head_ip):
        self.host_ids = host_ids
        self.host_info = host_info
        self.num_devices_per_host = num_devices_per_host
        self.num_devices = len(self.host_ids) * num_devices_per_host
        self.head_ip = head_ip

        self.server_address = f"{self.head_ip}:12345"
        self.service_server = None
        self.workers = []

    def launch_distributed_xla_service(self):
        self.service_server = xla_client._xla.get_distributed_runtime_service(
            self.server_address, len(self.host_ids))
        for i in range(len(self.host_ids)):
            node_resource = "node:" + self.host_info[i]["NodeManagerAddress"]
            cls = ray.remote(num_gpus=self.num_devices_per_host,
                             resources={node_resource:1e-3})(MeshHostWorker)
            self.workers.append(cls.remote(self.server_address, i))
        print("aha")

    def get_logical_device_mesh(self):
        device_id_mesh = np.arange(len(self.host_ids) * self.num_devices_per_host).\
            reshape(len(host_ids), self.num_devices_per_host)
        return LogicalDeviceMesh(device_id_mesh, [1, 1], [1, 0.01])

    def compile_hlo_module(self, hlo_module):
        for w in self.workers:
            w.compile.remote(hlo_module.as_serialized_hlo_module_proto())

    def execute(self, host_inputs):
        for i in range(len(self.workers)):
            self.workers[i].execute.remote(host_inputs[i])

    def sync_workers(self):
        ray.wait([w.sync.remote() for w in self.workers])


class MeshHostWorker:
    """A ray actor to manage the xla computation on a single node"""
    def __init__(self, server_address, node_id):
        self.node_id = 0
        self.client = xla_client._xla.get_distributed_runtime_client(server_address, node_id)
        self.client.connect()
        self.backend = xla_client._gpu_backend_factory(self.client, node_id=node_id)

    def compile(self, hlo_module_proto):
        backend = self.backend

        global_devices = backend.devices()
        global_device_ids = np.array(tuple(x.id for x in global_devices))

        num_replicas = len(global_device_ids)
        num_partitions = 1
        device_assignment = global_device_ids.reshape((num_replicas, num_partitions))
        device_assignment = xla_client.DeviceAssignment.create(device_assignment)
        use_spmd_partitioning = False

        compile_options = xla_client.CompileOptions()
        build_options = compile_options.executable_build_options
        build_options.num_replicas = num_replicas
        build_options.num_partitions = num_partitions
        build_options.use_spmd_partitioning = use_spmd_partitioning
        build_options.device_assignment = device_assignment

        computation = xla_client.XlaComputation(hlo_module_proto)
        compiled_computation = backend.compile(computation, compile_options)
        self.compiled_computation = compiled_computation

    def execute(self, device_inputs):
        local_devices = self.backend.local_devices()

        if isinstance(device_inputs[0][0], np.ndarray):
            device_inputs = [
                [self.backend.buffer_from_pyval(device_inputs[arg_id][device_id],
                                                local_devices[device_id])
                    for device_id in range(len(local_devices))]
                for arg_id in range(len(device_inputs))
            ]

        device_outs = self.compiled_computation.\
            execute_sharded_on_local_devices(device_inputs)
        for x in device_outs[0]:
            print("device_outs:", x, flush=True)

    def sync(self):
        return


class DeviceCluster:
    """A ray cluster with gpu devices"""
    def __init__(self, ray_address='auto'):
        self.head_info = ray.init(address=ray_address)
        self.head_ip = self.head_info['node_ip_address']

        # Gather hosts ids
        self.host_info = []
        for node in ray.nodes():
            found = False
            for key in node["Resources"]:
                if key.startswith("node:"):
                    self.host_info.append(node)

        # Gather device info
        self.num_devices = []
        for host_info in self.host_info:
            number = host_info["Resources"]["GPU"]
            assert number.is_integer()
            self.num_devices.append(int(number))

    def is_homogeneous(self, host_ids):
        for i in range(1, len(host_ids)):
            if self.num_devices[host_ids[0]] != self.num_devices[host_ids[i]]:
                return False
        return True

    def get_physical_mesh(self, host_ids=None):
        host_ids = host_ids or np.arange(len(self.host_info))
        host_info = [self.host_info[x] for x in host_ids]

        assert self.is_homogeneous(host_ids)

        return MultiHostDeviceMesh(host_ids, host_info, self.num_devices[host_ids[0]], self.head_ip)

