from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.lib import xla_client, xla_bridge

ops = xla_client.ops

MB = 1 << 20

def test_sin_cos():
    def f(x):
        return jax.numpy.sin(jax.numpy.cos(x.T))

    c = jax.xla_computation(f)(np.ones((10,8)))

    gpu_backend = xla_bridge.get_backend("gpu")
    compiled_computation = gpu_backend.compile(c)

    print(c.as_hlo_text())
    print(compiled_computation.hlo_modules()[0].to_string())

    host_input = np.ones((10,8), dtype=np.float32)
    device_input = gpu_backend.buffer_from_pyval(host_input)
    device_out = compiled_computation.execute([device_input,])


def parameter(builder, num, shape, dtype):
    shape = xla_client.Shape.array_shape(np.dtype(dtype), shape)
    name = ""
    replicated = []
    return ops.Parameter(builder, num,
                         shape.with_major_to_minor_layout_if_absent(), name,
                         replicated)

def test_alias():
    c = xla_client.XlaBuilder("test")
    a = parameter(c, 0, (8 * MB//4,), np.float32)
    b = parameter(c, 1, (8 * MB//4,), np.float32)
    d = parameter(c, 2, (8 * MB//4,), np.float32)
    e = parameter(c, 3, (8 * MB//4,), np.float32)

    backend = xla_bridge.get_backend("gpu")

    #z = ops.Add(a, b)
    z = ops.Constant(c, 0.1)

    #c.setup_alias((0,), 0, ())

    c = c.build(ops.Tuple(c, [z]))
    compiled_c = backend.compile(c)
    real_mem = compiled_c.total_allocation_size()

    print(compiled_c.hlo_modules()[0].to_string())
    print(f"{real_mem / MB:.2f} MB")

    #a = backend.buffer_from_pyval(np.ones((8 * MB // 4), dtype=np.float32))
    #b = backend.buffer_from_pyval(np.ones((8 * MB // 4), dtype=np.float32))
    #d = backend.buffer_from_pyval(np.ones((8 * MB // 4), dtype=np.float32))
    #e = backend.buffer_from_pyval(np.ones((8 * MB // 4), dtype=np.float32))

    #for i in range(10):
    #    ans, = compiled_c.execute([a, b, d, e])


def test_shard():
    c = xla_client.XlaBuilder("shard")
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions = [1]
    sharding.tile_assignment_devices = [0]
    c.set_sharding(sharding)
    x = ops.Parameter(c, 0, xla_client.shape_from_pyval(np.ones((10, 8), dtype=np.float32)))
    c.clear_sharding()
    y = ops.Parameter(c, 1, xla_client.shape_from_pyval(np.ones((10, 8), dtype=np.float32)))

    backend = xla_bridge.get_backend("gpu")

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

    backend = xla_bridge.get_backend("gpu")
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
    sharding.tile_assignment_dimensions = [2, 1]
    sharding.tile_assignment_devices = [0, 1]
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

    backend = xla_bridge.get_backend("gpu")
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
    sharding.tile_assignment_dimensions = [1, 1]
    sharding.tile_assignment_devices = [0,]
    c.set_sharding(sharding)
    x = parameter(c, 0, (2, 2), np.float32)

    z = ops.Add(x, x)
    z = ops.Add(z, z)
    z = ops.Add(z, z)
    c.clear_sharding()

    # Build a computational graph on device 1
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions = [1, 1]
    sharding.tile_assignment_devices = [1,]
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

    backend = xla_bridge.get_backend("gpu")
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


def test_reshard_multi_allgather():
    c = xla_client.XlaBuilder("shard")

    # Set input sharding
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions = [8, 2]
    sharding.tile_assignment_devices = list(range(16))
    c.set_sharding(sharding)
    x = parameter(c, 0, (32, 32), np.float32)
    c.clear_sharding()

    # Build computational graph
    y = ops.Constant(c, np.float32(1))
    z = ops.Broadcast(y, (32, 32))
    z = ops.Add(x, z)

    # Set output sharding
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.REPLICATED
    #sharding.tile_assignment_dimensions = [2, 2]
    ##sharding.replicate_on_last_tile_dim = True
    #sharding.tile_assignment_devices = [0, 1, 2, 3]

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
    num_partitions = 16
    use_spmd_partitioning = False
    device_assignment = xla_client.DeviceAssignment.create([list(range(num_partitions))])
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.num_partitions = num_partitions
    build_options.use_spmd_partitioning = True
    build_options.device_assignment = device_assignment

    backend = xla_bridge.get_backend("gpu")
    import alpa
    with alpa.XlaPassContext({
        "build_option::bypass_device_assignment_check": True,
    }):
        compiled_computation = backend.compile(c, compile_options)

    # Print spmd partitioned HLO
    print(compiled_computation.hlo_modules()[0].to_string())


def test_reshard_all_to_all():
    c = xla_client.XlaBuilder("shard")

    # Set input sharding
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions = [4, 1]
    sharding.tile_assignment_devices = list(range(4))
    c.set_sharding(sharding)
    x = parameter(c, 0, (32, 32), np.float32)
    c.clear_sharding()

    # Build computational graph
    if False:
        z = ops.Reshape(x, (2, 16, 32))
        sharding = xla_client.OpSharding()
        sharding.type = sharding.type.OTHER
        sharding.tile_assignment_dimensions = [2, 1, 2]
        sharding.tile_assignment_devices = list(range(4))
    else:
        z = x
        sharding = xla_client.OpSharding()
        sharding.type = sharding.type.OTHER
        sharding.tile_assignment_dimensions = [2, 2]
        sharding.tile_assignment_devicesi = list(range(4))

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
    num_partitions = 4
    use_spmd_partitioning = False
    device_assignment = xla_client.DeviceAssignment.create([list(range(num_partitions))])
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.num_partitions = num_partitions
    build_options.use_spmd_partitioning = True
    build_options.device_assignment = device_assignment

    backend = xla_bridge.get_backend("gpu")
    import alpa
    with alpa.XlaPassContext({
        "build_option::bypass_device_assignment_check": True,
    }):
        compiled_computation = backend.compile(c, compile_options)

    # Print spmd partitioned HLO
    print(compiled_computation.hlo_modules()[0].to_string())


def test_reshard_change_mesh_shape():
    c = xla_client.XlaBuilder("shard")

    # Set input sharding
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions = [1, 2, 2]
    sharding.tile_assignment_devices = [0, 1, 2, 3]
    sharding.replicate_on_last_tile_dim = True
    c.set_sharding(sharding)
    x = parameter(c, 0, (32, 32), np.float32)
    c.clear_sharding()

    # Build computational graph
    z = x
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions = [4, 1]
    sharding.tile_assignment_devices = [0, 1, 2, 3]

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
    num_partitions = 4
    use_spmd_partitioning = False
    device_assignment = xla_client.DeviceAssignment.create([list(range(num_partitions))])
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.num_partitions = num_partitions
    build_options.use_spmd_partitioning = True
    build_options.device_assignment = device_assignment

    backend = xla_bridge.get_backend("gpu")
    import alpa
    with alpa.XlaPassContext({
        "build_option::bypass_device_assignment_check": True,
    }):
        compiled_computation = backend.compile(c, compile_options)

    # Print spmd partitioned HLO
    print(compiled_computation.hlo_modules()[0].to_string())


def test_skip_hlo_passes():
    from alpa import XlaPassContext

    c = xla_client.XlaBuilder("shard")

    # Set input sharding
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.OTHER
    sharding.tile_assignment_dimensions = [2, 1]
    sharding.tile_assignment_devices = [0, 1]
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

    backend = xla_bridge.get_backend("gpu")
    with XlaPassContext({"build_option::skip_backend_codegen": True}):
        compiled_computation = backend.compile(c, compile_options)

    # Print spmd partitioned HLO
    hlo_module = compiled_computation.hlo_modules()[0]
    c = xla_client.XlaComputation(hlo_module.as_serialized_hlo_module_proto())

    with XlaPassContext({"build_option::skip_hlo_passes": True}):
        compiled_computation = backend.compile(c, compile_options)

    # Run
    host_input = np.ones((2, 2), dtype=np.float32)
    device_inputs = [[
        backend.buffer_from_pyval(host_input[[i],:], backend.devices()[i])
        for i in range(2)
    ]]
    device_outs = compiled_computation.execute_sharded_on_local_devices(device_inputs)
    print(device_outs)


def test_create_zero_buffers():
    shapes = ((2, 2), (3, 3))
    dtypes = (jnp.float32, jnp.float32)

    def compile_get_zero_buffers(backend, num_devices):
        c = xla_client.XlaBuilder("get_zero_buffers")
        sharding = xla_client.OpSharding()
        sharding.type = sharding.type.REPLICATED
        c.set_sharding(sharding)
        ret = []
        for shape, dtype in zip(shapes, dtypes):
            zero = ops.Constant(c, dtype(0))
            zero = ops.Broadcast(zero, shape)
            ret.append(zero)
        c.clear_sharding()
        c = c.build(ops.Tuple(c, ret))

        compile_options = xla_bridge.get_compile_options(
            num_replicas=1,
            num_partitions=num_devices,
            device_assignment=np.arange(num_devices).reshape((1, -1)),
            use_spmd_partitioning=True,
        )
        compiled_computation = backend.compile(c, compile_options)
        return compiled_computation

    backend = xla_bridge.get_backend("gpu")
    num_devices = 8
    get_zero_buffers = compile_get_zero_buffers(backend, num_devices)

    device_outs = get_zero_buffers.execute_sharded_on_local_devices([])

    print(device_outs)


if __name__ == "__main__":
    #test_sin_cos()
    #test_alias()
    #test_shard()

    #test_manual_construct_replica()
    #test_manual_construct_spmd_shard()
    #test_manual_construct_spmd_one_device()

    #test_reshard_multi_allgather()
    #test_reshard_all_to_all()
    test_reshard_change_mesh_shape()

    #test_skip_hlo_passes()

    #test_create_zero_buffers()

