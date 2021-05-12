from functools import partial
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.lib import xla_client, xla_bridge
import ray

from parax import LogicalDeviceMesh, XlaPassContext

ops = xla_client.ops


def parameter(builder, num, shape, dtype):
    shape = xla_client.Shape.array_shape(np.dtype(dtype), shape)
    name = ""
    replicated = []
    return ops.Parameter(builder, num,
                         shape.with_major_to_minor_layout_if_absent(), name,
                         replicated)


def all_reduce(builder, operand, reduce_op, replica_groups):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    if reduce_op == 'add':
        rc = xla_client.XlaBuilder("reduce_" + reduce_op)
        x = parameter(rc, 0, (), np.float32)
        y = parameter(rc, 1, (), np.float32)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    return ops.AllReduce(operand, rc, replica_groups_protos,
            None, None)


class DeviceCluster:
    def __init__(self, ray_address='auto'):
        self.head_info = ray.init(address=ray_address)
        self.head_ip = self.head_info['node_ip_address']

        # Gather hosts ids
        tmp_host_info = {}
        for node in ray.nodes():
            found = False
            for key in node["Resources"]:
                if key.startswith("parax_host"):
                    host_id = int(key.split("parax_host_")[1])
                    found = True
            assert found, "Please specify parax_host_id when launch ray runtime"
            tmp_host_info[host_id] = node

        self.host_info = [None] * len(tmp_host_info)
        for host_id, info in tmp_host_info.items():
            assert self.host_info[host_id] is None, "Found duplicated host id"
            self.host_info[host_id] = info

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

    def get_device_mesh(self, host_ids=None):
        host_ids = host_ids or np.arange(len(self.host_info))
        host_info = [self.host_info[x] for x in host_ids]

        assert self.is_homogeneous(host_ids)

        return DeviceMesh(host_ids, host_info, self.num_devices[host_ids[0]], self.head_ip)


class XlaHostWorker:
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


class MultiHostDeviceMesh:
    def __init__(self, host_ids, host_info, num_devices_per_host, head_ip):
        self.host_ids = host_ids
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
            cls = ray.remote(num_gpus=self.num_devices_per_host, resources={})(XlaHostWorker)
            self.workers.append(cls.remote(self.server_address, i))

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


def test_multi_host_all_reduce():
    device_cluster = DeviceCluster()

    print("Device mesh")
    device_mesh = device_cluster.get_device_mesh()

    def get_hlo_module_proto():
        backend = xla_client._gpu_backend_factory()
        c = xla_client.XlaBuilder("shard")
        x = parameter(c, 0, (5,), np.float32)
        z = all_reduce(c, x, 'add', (tuple(range(device_mesh.num_devices)),))
        c = c.build(ops.Tuple(c, [z]))

        global_device_ids = np.arange(device_mesh.num_devices)

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

        with XlaPassContext({
            "build_option::pass_through_device_assignment": True
        }):
            compiled_computation = backend.compile(c, compile_options)
        hlo_module = compiled_computation.hlo_modules()[0]
        return hlo_module

    # Prepare inputs. shape: (num_hosts, num_args, num_devices)
    dtype = np.float32
    host_inputs = [   
        [[np.ones(5, dtype=dtype), np.ones(5, dtype=dtype)]],
        [[np.ones(5, dtype=dtype), np.ones(5, dtype=dtype)]],
    ]

    # Compile and run
    hlo_module = get_hlo_module_proto()
    device_mesh.launch_distributed_xla_service()
    device_mesh.compile_hlo_module(hlo_module)
    device_mesh.execute(host_inputs)
    device_mesh.sync_workers()


def test_multi_host_auto_sharding():
    device_cluster = DeviceCluster()

    print("Device mesh")
    device_mesh = device_cluster.get_device_mesh()

    def get_hlo_module_proto():
        @parallelize(devices=device_mesh)
        def add_one(x):
            x = x + 1
            return x


    # Prepare inputs. shape: (num_hosts, num_args, num_devices)
    dtype = np.float32
    host_inputs = [   
        [[np.ones(5, dtype=dtype), np.ones(5, dtype=dtype)]],
        [[np.ones(5, dtype=dtype), np.ones(5, dtype=dtype)]],
    ]

    # Compile and run
    hlo_module = get_hlo_module_proto()
    device_mesh.launch_distributed_xla_service()
    device_mesh.compile_hlo_module(hlo_module)
    device_mesh.execute(host_inputs)
    device_mesh.sync_workers()


if __name__ == "__main__":
    #test_multi_host_all_reduce()
    test_multi_host_auto_sharding()


