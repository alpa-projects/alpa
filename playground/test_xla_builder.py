from functools import partial

import numpy as np
import jax
from jax.lib import xla_client
import jax.numpy as jnp
from jax.lib import xla_bridge

ops = xla_client.ops


def test_sin_cos():
    def f(x):
        return jax.numpy.sin(jax.numpy.cos(x.T))

    c = jax.xla_computation(f)(np.ones((10,8)))

    gpu_backend = xla_client.get_local_backend("gpu")
    compiled_computation = gpu_backend.compile(c)

    print(c.as_hlo_text())
    print(compiled_computation.hlo_modules()[0].to_string())

    host_input = np.ones((10,8), dtype=np.float32)
    device_input = gpu_backend.buffer_from_pyval(host_input)
    device_out = compiled_computation.execute([device_input,])


def test_shard():
    c = xla_client.XlaBuilder("shard")
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions.extend([1])
    sharding.tile_assignment_devices.extend([0])
    c.set_sharding(sharding)
    x = ops.Parameter(c, 0, xla_client.shape_from_pyval(np.ones((10, 8), dtype=np.float32)))
    c.clear_sharding()
    y = ops.Parameter(c, 1, xla_client.shape_from_pyval(np.ones((10, 8), dtype=np.float32)))

    backend = xla_client.get_local_backend("gpu")

    z = ops.Add(x, y)
    z = ops.Add(z, y)

    c = c.build(z)
    #print(c.as_hlo_text())

    compiled_c = backend.compile(c)

    print(compiled_c.hlo_modules()[0].to_string())

    x = backend.buffer_from_pyval(np.ones((10, 8), dtype=np.float32))
    y = backend.buffer_from_pyval(np.ones((10, 8), dtype=np.float32))
    ans, = compiled_c.execute([x, y])


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
    z = all_reduce(c, z, 'add', ((0, 1, 2, 3,),))

    c = c.build(ops.Tuple(c, [z]))
    print(c.as_hlo_text())

    num_replicas = 4
    num_partitions = 1
    device_assignment = xla_client.DeviceAssignment.create([[0], [1], [2], [3]])
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
        for i in range(4)
    ]]

    device_outs = compiled_computation.execute_sharded_on_local_devices(device_inputs)
    print(device_outs)


def test_manual_construct_spmd_shard():
    c = xla_client.XlaBuilder("shard")

    # Set input sharding
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions.extend([2, 1])
    sharding.tile_assignment_devices.extend([0, 1])
    c.set_sharding(sharding)
    x = parameter(c, 0, (2, 2), np.float32)
    c.clear_sharding()

    # Build computational graph
    y = ops.Constant(c, np.float32(1))
    z = ops.Broadcast(y, (2, 2))
    z = ops.Add(x, z)

    # Set output sharding
    sharding2 = xla_client.OpSharding()
    sharding2.type = sharding.type.TUPLE
    sharding2.tuple_shardings = [sharding]
    c.set_sharding(sharding2)
    out = ops.Tuple(c, [z])
    c.clear_sharding()

    # Build HLO
    c = c.build(out)
    print(c.as_hlo_text())
    print("=" * 20)

    # Compile
    num_replicas = 1
    num_partitions = 2
    use_spmd_partitioning = False
    device_assignment = xla_client.DeviceAssignment.create([[0, 1]])
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.num_partitions = num_partitions
    build_options.use_spmd_partitioning = True
    build_options.device_assignment = device_assignment

    backend = xla_client.get_local_backend("gpu")
    compiled_computation = backend.compile(c, compile_options)

    # Print spmd partitioned HLO
    print(compiled_computation.hlo_modules()[0].to_string())

    # Run
    host_input = np.ones((2, 2), dtype=np.float32)
    device_inputs = [[
        backend.buffer_from_pyval(host_input[[i],:], backend.devices()[i])
        for i in range(2)
    ]]
    device_outs = compiled_computation.execute_sharded_on_local_devices(device_inputs)
    print(device_outs)


def test_manual_construct_spmd_one_device():
    c = xla_client.XlaBuilder("shard")

    # Build a computational graph on device 0
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions.extend([1, 1])
    sharding.tile_assignment_devices.extend([0,])
    c.set_sharding(sharding)
    x = parameter(c, 0, (2, 2), np.float32)

    z = ops.Add(x, x)
    z = ops.Add(z, z)
    z = ops.Add(z, z)
    c.clear_sharding()

    # Build a computational graph on device 1
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions.extend([1, 1])
    sharding.tile_assignment_devices.extend([1,])
    c.set_sharding(sharding)
    z = ops.Add(z, z)
    z = ops.Add(z, z)
    out = z
    c.clear_sharding()

    # Build HLO
    c = c.build(out)
    print(c.as_hlo_text())
    print("=" * 20)

    # Compile
    num_replicas = 1
    num_partitions = 2
    use_spmd_partitioning = False
    device_assignment = xla_client.DeviceAssignment.create([[0, 1]])
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.num_partitions = num_partitions
    build_options.use_spmd_partitioning = True
    build_options.device_assignment = device_assignment

    backend = xla_client.get_local_backend("gpu")
    compiled_computation = backend.compile(c, compile_options)

    # Print spmd partitioned HLO
    print(compiled_computation.hlo_modules()[0].to_string())

    # Run
    host_input = np.ones((2, 2), dtype=np.float32)
    device_inputs = [[
        backend.buffer_from_pyval(host_input, backend.devices()[0]),
        backend.buffer_from_pyval(host_input, backend.devices()[1]),
    ]]
    device_outs = compiled_computation.execute_sharded_on_local_devices(device_inputs)
    print(device_outs)


if __name__ == "__main__":
    #test_sin_cos()
    #test_shard()

    #test_manual_construct_replica()
    #test_manual_construct_spmd_shard()
    test_manual_construct_spmd_one_device()

