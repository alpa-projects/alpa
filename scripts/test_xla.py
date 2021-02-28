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

    print(c.as_hlo_text())

    gpu_backend = xla_client.get_local_backend("gpu")
    compiled_computation = gpu_backend.compile(c)

    host_input = np.ones((10,8), dtype=np.float32)
    device_input = gpu_backend.buffer_from_pyval(host_input)
    device_out = compiled_computation.execute([device_input ,])

    print(type(c))
    print(type(compiled_computation))


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
    #print(dir(ans))


def test_pmap_1d():
    def f(x):
        return x + 1

    x = jnp.ones((4, 10))
    parallel_f = jax.pmap(f, in_axes=0, out_axes=0)

    z = parallel_f(x)


def test_matmul_k_partition():
    def matmul_k_partition(lhs, rhs):
        @partial(jax.pmap,
                 axis_name='k',
                 in_axes=(1, 0),
                 out_axes=None)
        def matmul(lhs, rhs):
            res = lhs @ rhs
            return jax.lax.psum(res, axis_name='k')

        return matmul(lhs, rhs)

    a = jnp.ones((1024, 4, 256))
    b = jnp.ones((4, 256, 1024))

    c = matmul_k_partition(a, b)

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
    y = ops.Constant(c, np.float32(0))
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


def test_manual_construct_spmd():
    c = xla_client.XlaBuilder("shard")
    x = parameter(c, 0, (2, 2), np.float32)
    y = ops.Constant(c, np.float32(0))
    z = ops.Broadcast(y, (2, 2))
    z = ops.Add(x, z)
    z = all_reduce(c, z, 'add', ((0, 1,),))

    c = c.build(ops.Tuple(c, [z]))
    print(c.as_hlo_text())

    num_replicas = 1
    num_partitions = 2
    device_assignment = xla_client.DeviceAssignment.create([[0, 1]])
    use_spmd_partitioning = False

    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.num_partitions = num_partitions
    build_options.use_spmd_partitioning = True
    build_options.device_assignment = device_assignment

    backend = xla_client.get_local_backend("gpu")
    compiled_computation = backend.compile(c, compile_options)

    host_input = np.ones((2, 2), dtype=np.float32)
    device_inputs = [[
        backend.buffer_from_pyval(host_input, backend.devices()[i])
        for i in range(2)
    ]]

    device_outs = compiled_computation.execute_sharded_on_local_devices(device_inputs)
    print(device_outs)


if __name__ == "__main__":
    #test_sin_cos()
    #test_shard()

    #test_pmap_1d()
    #test_matmul_k_partition()
    test_manual_construct_replica()
    #test_manual_construct_spmd()

