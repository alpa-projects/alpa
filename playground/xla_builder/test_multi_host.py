from functools import partial

import numpy as np
import jax
from jax.lib import xla_client
import jax.numpy as jnp
from jax.lib import xla_bridge

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


def test_manual_construct_replica():
    c = xla_client.XlaBuilder("shard")
    x = parameter(c, 0, (2, 2), np.float32)
    y = ops.Constant(c, np.float32(1))
    z = ops.Broadcast(y, (2, 2))
    z = ops.Add(x, z)
    z = all_reduce(c, z, 'add', ((0, 1,),))

    c = c.build(ops.Tuple(c, [z]))
    print(c.as_hlo_text())

    num_replicas = 2
    num_partitions = 1
    device_assignment = xla_client.DeviceAssignment.create([[0], [1]])
    use_spmd_partitioning = False

    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.num_partitions = num_partitions
    build_options.use_spmd_partitioning = use_spmd_partitioning
    build_options.device_assignment = device_assignment

    backend = xla_client.get_local_backend("gpu")
    compiled_computation = backend.compile(c, compile_options)

    host_input = np.ones((2,2), dtype=np.float32)
    device_inputs = [[
        backend.buffer_from_pyval(host_input, backend.devices()[i])
        for i in range(num_replicas)
    ]]

    device_outs = compiled_computation.execute_sharded_on_local_devices(device_inputs)
    for x in device_outs[0]:
        print(x)

if __name__ == "__main__":
    test_manual_construct_replica()

