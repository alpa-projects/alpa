from functools import partial
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.lib import xla_client, xla_bridge
import ray

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


@ray.remote
class Server:
    def __init__(self, port, num_nodes):
        self.server = xla_client._xla.get_distributed_runtime_service(f"[::0]:{port}", num_nodes)


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, port, node_id):
        self.node_id = 0
        self.client = xla_client._xla.get_distributed_runtime_client(f"dns:///localhost:{port}", node_id)
        self.client.connect()
        self.backend = xla_client._gpu_backend_factory(self.client, node_id=node_id)   # PyClient

    def compile(self):
        c = xla_client.XlaBuilder("shard")
        x = parameter(c, 0, (5,), np.float32)
        z = all_reduce(c, x, 'add', (()))
        c = c.build(ops.Tuple(c, [z]))

        backend = self.backend
        local_devices = backend.local_devices()
        global_devices = backend.devices()
        local_device_ids = np.array([x.id for x in local_devices])
        global_device_ids = np.array([x.id for x in global_devices])

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
    
        compiled_computation = backend.compile(c, compile_options)
        hlo_module = compiled_computation.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()
        #print(hlo_ir, flush=True)
    
        host_input = np.ones((5,), dtype=np.float32)
        device_inputs = [[
            backend.buffer_from_pyval(host_input, local_devices[i])
            for i in range(len(local_devices))
        ]]
        print("device_inputs:", device_inputs[0], flush=True)
    
        device_outs = compiled_computation.execute_sharded_on_local_devices(device_inputs)
        for x in device_outs[0]:
            print("device_outs:", x, flush=True)


def test_multi_host():
    ray.init()

    port = 8485
    num_nodes = 2

    server = Server.remote(port, num_nodes)
    workers = []
    for i in range(num_nodes):
        workers.append(Worker.remote(port, i))

    tasks = []
    for i in range(len(workers)):
        tasks.append((workers[i].compile.remote()))

    ray.wait(tasks)


if __name__ == "__main__":
    test_multi_host()

