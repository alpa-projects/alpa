"""Cluster related configurations (e.g., topology)."""

import numpy as np
import ray
from jax.lib import xla_client, xla_bridge

from parax.util import get_dim_last_value, to_int_tuple
from parax.xla_pass_context import XlaPassContext


class LogicalDeviceMesh:
    """
    A logical multi-dimensional device mesh.

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
    """A physical device mesh that presents devices on a single node."""

    def __init__(self, devices):
        self.devices = devices

    def get_logical_mesh(self, mesh_shape, mesh_alpha, mesh_beta):
        """Get a mapping to logoical mesh."""
        device_ids = np.array([d.id for d in self.devices])
        device_ids = device_ids.reshape(mesh_shape)
        return LogicalDeviceMesh(self, device_ids, mesh_alpha, mesh_beta)

    def get_default_logical_mesh(self):
        return self.get_logical_mesh((1, len(self.devices)), [1, 1], [1, 1])


class MultiHostDeviceMesh:
    """A physical device mesh that presents a device mesh on multipe nodes."""

    def __init__(self, host_ids, host_info, num_devices_per_host, head_ip):
        self.host_ids = host_ids
        self.host_info = host_info
        self.num_devices_per_host = num_devices_per_host
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
                             resources={node_resource: 1e-3})(MeshHostWorker)
            self.workers.append(cls.remote(self.server_address, i))

    def get_logical_mesh(self, mesh_shape, mesh_alpha, mesh_beta):
        """Get a mapping to logoical mesh."""
        id_mesh = np.arange(len(self.host_ids) * self.num_devices_per_host).\
            reshape(mesh_shape)
        return LogicalDeviceMesh(self, id_mesh, mesh_alpha, mesh_beta)

    def get_default_logical_mesh(self):
        return self.get_logical_mesh((len(self.host_ids), self.num_devices_per_host),
            [1, 1], [1, 0.01])

    def compile_hlo_module(self,
                           hlo_proto,
                           logical_mesh_shape,
                           auto_sharding_strategy_vector,
                           is_tuple_args):
        for w in self.workers:
            w.compile_hlo_module.remote(
                hlo_proto,
                logical_mesh_shape,
                auto_sharding_strategy_vector,
                is_tuple_args)

    def execute(self, host_inputs):
        for i in range(len(self.workers)):
            self.workers[i].execute.remote(host_inputs[i])

    def sync_workers(self):
        ray.wait([w.sync.remote() for w in self.workers])


class MeshHostWorker:
    """A ray actor to manage the xla computation on a single host."""

    def __init__(self, server_address, node_id):
        self.node_id = 0
        self.client = xla_client._xla.get_distributed_runtime_client(server_address, node_id)
        self.client.connect()
        self.backend = xla_client._gpu_backend_factory(self.client, node_id=node_id)
        self.compiled_computation = None

    def compile_hlo_module(self,
                           hlo_proto,
                           logical_mesh_shape,
                           auto_sharding_strategy_vector,
                           is_tuple_args):
        backend = self.backend
        num_devices = np.prod(logical_mesh_shape)

        assert num_devices == len(backend.devices())

        device_ids = np.arange(num_devices)
        compile_options = xla_bridge.get_compile_options(
            num_replicas=1,
            num_partitions=len(device_ids),
            device_assignment=device_ids.reshape((1, len(device_ids))),
            use_spmd_partitioning=True,
        )
        compile_options.parameter_is_tupled_arguments = is_tuple_args

        computation = xla_client.XlaComputation(hlo_proto)
        with XlaPassContext({
            "auto_sharding::enable": True,
            "auto_sharding::load_strategy": True,
            "auto_sharding::strategy_vector": to_int_tuple(auto_sharding_strategy_vector),

            # Device mesh
            "auto_sharding::device_mesh_ids": to_int_tuple(device_ids),
            "auto_sharding::device_mesh_shape": tuple(logical_mesh_shape),

            # Other useless but required arguments
            "auto_sharding::device_mesh_alpha": (1.0,) * len(logical_mesh_shape),
            "auto_sharding::device_mesh_beta": (1.0,) * len(logical_mesh_shape),
        }):
            compiled_computation = backend.compile(computation, compile_options)

        #hlo_module = compiled_computation.hlo_modules()[0]

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

    def sync(self):
        return


class DeviceCluster:
    """A ray cluster with gpu devices."""

    def __init__(self, ray_address='auto'):
        self.head_info = ray.init(address=ray_address)
        self.head_ip = self.head_info['node_ip_address']

        # Gather host ids
        self.host_info = []
        for node in ray.nodes():
            for key in node["Resources"]:
                if key.startswith("node:"):
                    self.host_info.append(node)

        # Gather device info
        self.num_devices = []
        for host_info in self.host_info:
            number = host_info["Resources"]["GPU"]
            assert number.is_integer()
            self.num_devices.append(int(number))

    def get_physical_mesh(self, host_ids=None, num_devices_per_host=None):
        """
        Slice a subset of hosts and devices to form a physical device mesh.

        Args:
            host_ids: List[int]. The index of host nodes.
                'None' means using all hosts
            num_devices_per_host: int. The number of devices per host.
                'None' means using all devices

        Return:
            A physical multi-host device mesh
        """
        host_ids = host_ids or np.arange(len(self.host_info))
        host_info = [self.host_info[x] for x in host_ids]

        num_devices_per_host = num_devices_per_host or self.num_devices[host_ids[0]]
        for host_id in host_ids:
            assert self.num_devices[host_id] >= num_devices_per_host

        return MultiHostDeviceMesh(host_ids, host_info, num_devices_per_host, self.head_ip)
