# pylint: disable=no-self-use,consider-using-enumerate
"""Profiling communication cost."""
from collections import defaultdict
import time

import numpy as np

from jax.lib import xla_client, xla_bridge

from parax.util import GB

ops = xla_client.ops


class ProfilingResult:
    """Store the profiling result."""

    def __init__(self):
        # Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
        # assume the elements in the list is sorted according to the size (ascending).
        self.all_reduce_cost_dict = defaultdict(list)
        self.all_gather_cost_dict = defaultdict(list)
        self.reduce_scatter_cost_dict = defaultdict(list)

    def record_all_reduce(self, group, size, dtype, time_cost):
        key = (group, dtype)
        self.all_reduce_cost_dict[key].append((size, time_cost))

    def record_all_gather(self, group, size, dtype, time_cost):
        key = (group, dtype)
        self.all_gather_cost_dict[key].append((size, time_cost))

    def estimate_all_reduce(self, group, size, dtype):
        ret = self._estimate_internal(group, size, dtype, self.all_reduce_cost_dict) -\
            self._estimate_internal(group, 0, dtype, self.all_reduce_cost_dict)
        return ret

    def estimate_all_gather(self, group, size, dtype):
        ret = self._estimate_internal(group, size, dtype, self.all_gather_cost_dict) -\
            self._estimate_internal(group, 0, dtype, self.all_gather_cost_dict)
        return ret

    def multiply_scale(self, factor):
        """
        Multiply the time cost by a constant factor.

        This is used to make the scale of time cost similar to the old alpha-beta model.
        """
        self._multiply_scale_internal(factor, self.all_reduce_cost_dict)
        self._multiply_scale_internal(factor, self.all_gather_cost_dict)
        self._multiply_scale_internal(factor, self.reduce_scatter_cost_dict)

    def make_monotonic(self):
        """Make the bandwidth monotonically increase along with the communication size."""
        self._make_monotonic_internal(self.all_reduce_cost_dict)
        self._make_monotonic_internal(self.all_gather_cost_dict)
        self._make_monotonic_internal(self.reduce_scatter_cost_dict)

    def _make_monotonic_internal(self, cost_dict):
        new_cost_dict = {}

        for key, value in cost_dict.items():
            # make bandwidth monotonically increasing
            sizes = np.array([x[0] for x in value])
            times = np.array([x[1] for x in value])
            bandwidth = sizes / times
            for i in range(1, len(bandwidth)):
                bandwidth[i] = max(bandwidth[i], bandwidth[i - 1])

            new_times = np.empty_like(times)
            for i in range(len(times)):
                if sizes[i] == 0 or bandwidth[i] == 0:
                    new_times[i] = value[i][1]
                else:
                    new_times[i] = sizes[i] / bandwidth[i]

            new_value = [(value[i][0], new_times[i]) for i in range(len(value))]
            new_cost_dict[key] = new_value

        cost_dict.update(new_cost_dict)

    def _multiply_scale_internal(self, factor, cost_dict):
        for value in cost_dict.values():
            for i in range(len(value)):
                value[i] = (value[i][0], value[i][1] * factor)

    def _estimate_internal(self, group, size, dtype, cost_dict):
        key = (group, dtype)
        cost_list = cost_dict[key]
        assert cost_list, f"Cannot find records for {(group, dtype)}"

        if size > cost_list[-1][0]:
            i = len(cost_list) - 2
        elif size < cost_list[0][0]:
            i = 0
        else:
            for i in range(len(cost_list) - 1):
                if cost_list[i][0] <= size <= cost_list[i + 1][0]:
                    break

        left_size = cost_list[i][0]
        left_cost = cost_list[i][1]
        right_size = cost_list[i + 1][0]
        right_cost = cost_list[i + 1][1]

        return (size - left_size) / (right_size - left_size) * (right_cost - left_cost) + left_cost

    def __str__(self):
        ret = "=== all_reduce_cost_dict ===\n"
        for key, value in self.all_reduce_cost_dict.items():
            num_devices = len(key[0][0])
            sizes = np.array([x[0] for x in value])
            times = np.array([x[1] for x in value])
            comm_bytes = 2 * (num_devices - 1) / num_devices * sizes * np.dtype(key[1]).itemsize
            bandwidth = comm_bytes / times / GB
            bandwidth_str = ", ".join(f"{x:.2f}" for x in bandwidth)
            ret += f"Key: {key}\nBandwidth: {bandwidth_str}\n"
        return ret


def _op_parameter(builder, num, shape, dtype):
    shape = xla_client.Shape.array_shape(np.dtype(dtype), shape)
    name = ""
    replicated = []
    return ops.Parameter(builder, num,
                         shape.with_major_to_minor_layout_if_absent(), name,
                         replicated)


def _op_all_reduce(builder, operand, dtype, reduce_op, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    if reduce_op == 'add':
        rc = xla_client.XlaBuilder("reduce_" + reduce_op)
        x = _op_parameter(rc, 0, (), dtype)
        y = _op_parameter(rc, 1, (), dtype)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    ret = ops.AllReduce(operand, rc, replica_groups_protos,
                        channel_id, None, True)
    return ret


def _op_all_gather(builder, operand, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    ret = ops.AllGather(operand, 0, len(replica_groups[0]),
                        replica_groups_protos, channel_id, None, True)
    return ret


def compile_collective_hlo(backend, num_devices, replica_groups, shape, dtype, primitive_name):
    """
    Compile a xla executable for benchmarking collective communication primitives.

    It is a while loop that calls the collective primitive for multiple times.
    """
    if primitive_name == "all-reduce":
        in_shape = out_shape = shape
    elif primitive_name == "all-gather":
        in_shape = (shape[0] // len(replica_groups[0]),)
        out_shape = shape
    else:
        raise ValueError("Invalid primitive: " + primitive_name)

    in_tuple_shape = xla_client.Shape.tuple_shape([
        xla_client.Shape.array_shape(np.dtype(dtype), in_shape),
        xla_client.Shape.array_shape(np.dtype(dtype), out_shape),
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
    if primitive_name == "all-reduce":
        out_buf = _op_all_reduce(body, in_buf, dtype, "add", replica_groups, channel_id)
    elif primitive_name == "all-gather":
        if in_shape[0] == 0 or out_shape[0] == 0:
            pass
        else:
            out_buf = _op_all_gather(body, in_buf, replica_groups, channel_id)
    else:
        raise ValueError("Invalid primitive: " + primitive_name)
    counter = ops.Sub(counter, ops.Constant(body, np.int32(1)))
    body.clear_sharding()
    ops.Tuple(body, [in_buf, out_buf, counter])
    body_computation = body.build()

    # condition
    cond = xla_client.XlaBuilder("condition")
    in_tuple = ops.Parameter(cond, 0, in_tuple_shape)
    counter = ops.GetTupleElement(in_tuple, 2)
    ops.Gt(counter, ops.Constant(cond, np.int32(0)))
    cond_computation = cond.Build()

    # while loop
    loop = xla_client.XlaBuilder("loop")
    in_buf = _op_parameter(loop, 0, in_shape, dtype)
    out_buf = _op_parameter(loop, 1, out_shape, dtype)
    counter = _op_parameter(loop, 2, (), np.int32)
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

    return in_shape, out_shape, backend.compile(loop_computation, compile_options)


def profile_collective_one_config(shape, dtype, replica_groups, primitive_name,
                                  backend, num_devices, local_devices,
                                  distributed_client, host_id, sync_func,
                                  number=10, warmup=2):
    """Profile the time cost of a collective communication primitive."""
    in_shape, out_shape, compiled = compile_collective_hlo(
        backend, num_devices, replica_groups, shape, dtype, primitive_name)
    xla_client._xla.init_nccl_communicators(backend, distributed_client,
                                            host_id, compiled)

    #real_mem = compiled.total_allocation_size()
    #print(compiled.hlo_modules()[0].to_string())
    #print(f"{real_mem / GB:.3f} GB")

    # Warm up
    device_inputs = [
        [backend.buffer_from_pyval(np.empty(in_shape, dtype), local_devices[i])
            for i in range(len(local_devices))],
        [backend.buffer_from_pyval(np.empty(out_shape, dtype), local_devices[i])
            for i in range(len(local_devices))],
        [backend.buffer_from_pyval(np.int32(warmup), local_devices[i])
            for i in range(len(local_devices))]
    ]
    device_inputs = compiled.execute_sharded_on_local_devices(device_inputs)

    # Run profiling
    device_inputs[2] = \
        [backend.buffer_from_pyval(np.int32(number), local_devices[i])
            for i in range(len(local_devices))]

    sync_func()
    tic = time.time()
    compiled.execute_sharded_on_local_devices(device_inputs)
    sync_func()
    toc = time.time()

    return (toc - tic) / number
