"""Profiling communication cost."""
from collections import defaultdict

import numpy as np

from jax.lib import xla_client, xla_bridge

ops = xla_client.ops


class ProfilingResult:
    def __init__(self):
        # Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
        # assume the elements in the list is sorted according to the size (ascending).
        self.all_reduce_cost = defaultdict(list)

    def serialize():
        keys = []
        lens = []
        values = []

        for key, value in self.all_reduce_cost.items():
            pass

    def record_all_reduce(self, group, size, dtype, time_cost):
        key = (group, dtype)
        self.all_reduce_cost[key].append((size, time_cost))

    def estimate_all_reduce(self, group, size, dtype):
        ret = self._estimate_all_reduce_internal(group, size, dtype) -\
              self._estimate_all_reduce_internal(group, 0, dtype)
        return ret

    def _estimate_all_reduce_internal(self, group, size, dtype):
        key = (group, dtype)
        l = self.all_reduce_cost[key]
        assert l

        if size > l[-1][0]:
            i = len(l) - 2
        elif size < l[0][0]:
            i = 0
        else:
            for i in range(len(l)-1):
                if l[i][0] <= size <= l[i+1][0]:
                    break

        left_size = l[i][0]
        left_cost = l[i][1]
        right_size = l[i+1][0]
        right_cost = l[i+1][1]

        return (size - left_size) / (right_size - left_size) * (right_cost - left_cost) + left_cost

    def __str__(self):
        ret = ""
        for key, value in self.all_reduce_cost.items():
            ret += str(key) + "\n"
            ret += str(value) + "\n"
        return ret


def op_parameter(builder, num, shape, dtype):
    shape = xla_client.Shape.array_shape(np.dtype(dtype), shape)
    name = ""
    replicated = []
    return ops.Parameter(builder, num,
                         shape.with_major_to_minor_layout_if_absent(), name,
                         replicated)


def op_all_reduce(builder, operand, reduce_op, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    if reduce_op == 'add':
        rc = xla_client.XlaBuilder("reduce_" + reduce_op)
        x = op_parameter(rc, 0, (), np.float32)
        y = op_parameter(rc, 1, (), np.float32)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    ret = ops.AllReduce(operand, rc, replica_groups_protos,
            channel_id, None, True)
    return ret


def compile_all_reduce(backend, num_devices, shape, dtype, reduce_op, replica_groups):
    """Compile a Runnable to benchmark the time cost of all-reduce"""
    in_tuple_shape = xla_client.Shape.tuple_shape([
        xla_client.Shape.array_shape(np.dtype(dtype), shape),
        xla_client.Shape.array_shape(np.dtype(dtype), shape),
        xla_client.Shape.array_shape(np.dtype(np.int32), ()),
    ])

    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions.extend([1])
    sharding.tile_assignment_devices.extend([0])

    # body
    body = xla_client.XlaBuilder("body")
    in_tuple = ops.Parameter(body, 0, in_tuple_shape)
    in_buf = ops.GetTupleElement(in_tuple, 0)
    out_buf = ops.GetTupleElement(in_tuple, 1)
    counter = ops.GetTupleElement(in_tuple, 2)
    channel_id = backend.create_channel_handle()
    body.set_sharding(sharding)
    out_buf = op_all_reduce(body, in_buf, reduce_op, replica_groups, channel_id)
    counter = ops.Sub(counter, ops.Constant(body, np.int32(1)))
    body.clear_sharding()
    ops.Tuple(body, [in_buf, out_buf, counter])
    body_computation = body.build()

    # condition
    cond = xla_client.XlaBuilder("condition")
    in_tuple = ops.Parameter(cond, 0, in_tuple_shape)
    counter = ops.GetTupleElement(in_tuple, 2)
    pred = ops.Gt(counter, ops.Constant(cond, np.int32(0)))
    cond_computation = cond.Build()

    # while loop
    loop = xla_client.XlaBuilder("loop")
    in_buf = op_parameter(loop, 0, shape, dtype)
    out_buf = op_parameter(loop, 1, shape, dtype)
    counter = op_parameter(loop, 2, (), np.int32)
    while_init = ops.Tuple(loop, [in_buf, out_buf, counter])
    ops.While(cond_computation, body_computation, while_init)
    loop.setup_alias((0,), 0, ())
    loop.setup_alias((1,), 1, ())
    loop_computation = loop.Build()

    compile_options = xla_bridge.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )

    return backend.compile(loop_computation, compile_options)

