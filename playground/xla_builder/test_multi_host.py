import numpy as np
import ray
from jax.lib import xla_client

from alpa import DeviceCluster, XlaPassContext, parallelize, global_config

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


def test_multi_host_all_reduce():
    device_cluster = DeviceCluster()

    print("Device mesh")
    device_mesh = device_cluster.get_physical_mesh()

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
    device_mesh.compile_hlo_module(hlo_module, None, None)
    device_mesh.execute(host_inputs)
    device_mesh.sync_workers()


def test_multi_host_auto_sharding():
    global_config.shard_parallel_strategy = "auto_sharding"

    device_cluster = DeviceCluster()
    physical_mesh = device_cluster.get_physical_mesh()
    num_devices = len(physical_mesh.host_ids) * physical_mesh.num_devices_per_host
    logical_mesh = physical_mesh.get_logical_mesh([1, num_devices], [1, 1], [1, 1])

    @parallelize(devices=logical_mesh)
    def add_one(x):
        x = x + 1
        return x

    a = np.ones((1000, 1000))
    out = add_one(a)

    print("Output", out)


if __name__ == "__main__":
    ray.init(address="auto")
    test_multi_host_auto_sharding()
