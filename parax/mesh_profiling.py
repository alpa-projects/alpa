# pylint: disable=no-self-use
"""Profiling communication cost."""
from collections import defaultdict
import time

import numpy as np

from jax.lib import xla_client, xla_bridge

from parax.util import GB, print_used_time

ops = xla_client.ops


class ProfilingResult:
    """Store the profiling result."""

    def __init__(self):
        # Cost dictionary for communication primitives
        # Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
        # The elements in the list is sorted according to the size (ascending).
        self.all_reduce_cost_dict = defaultdict(list)
        self.all_gather_cost_dict = defaultdict(list)
        self.reduce_scatter_cost_dict = defaultdict(list)
        self.all_to_all_cost_dict = defaultdict(list)

        # Cost dictionary for computation primitives
        # Dict[dtype] -> List[Tuple(flop_count, flop_per_second)]
        # The elements in the list is sorted according to the size (ascending).
        self.matmul_cost_dict = []
        self.conv_cost_dict = []

        # Cost dictionary for specific operators
        # Dict[op_info] -> double
        self.op_cost_dict = []

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

        return (size - left_size) / (right_size - left_size) * (
            right_cost - left_cost) + left_cost

    def __str__(self):
        ret = "=== all_reduce_cost_dict ===\n"
        for key, value in self.all_reduce_cost_dict.items():
            num_devices = len(key[0][0])
            sizes = np.array([x[0] for x in value])
            times = np.array([x[1] for x in value])
            comm_bytes = 2 * (num_devices - 1) / num_devices * sizes * np.dtype(
                key[1]).itemsize
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


def _op_all_reduce(operand, dtype, reduce_op, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    if reduce_op == 'add':
        rc = xla_client.XlaBuilder("reduce_" + reduce_op)
        x = _op_parameter(rc, 0, (), dtype)
        y = _op_parameter(rc, 1, (), dtype)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    ret = ops.AllReduce(operand, rc, replica_groups_protos, channel_id, None,
                        True)
    return ret


def _op_reduce_scatter(operand, dtype, reduce_op, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    if reduce_op == 'add':
        rc = xla_client.XlaBuilder("reduce_" + reduce_op)
        x = _op_parameter(rc, 0, (), dtype)
        y = _op_parameter(rc, 1, (), dtype)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    ret = ops.ReduceScatter(operand, rc, 0, len(replica_groups[0]),
                            replica_groups_protos, channel_id, None, True)
    return ret


def _op_all_gather(operand, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    ret = ops.AllGather(operand, 0, len(replica_groups[0]),
                        replica_groups_protos, channel_id, None, True)
    return ret


def _op_all_to_all(operand, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    ret = ops.AllToAll(operand, 0, 0, len(replica_groups[0]),
                       replica_groups_protos, channel_id, None, True)
    return ret


def compile_profiling_executable(backend, shapes, op_func, num_devices):
    """
    Compile a xla executable for benchmarking operators.
    It is a while loop that calls the operator for multiple times.
    """

    in_tuple_shape = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(np.dtype(np.int32), ())] + [
            xla_client.Shape.array_shape(np.dtype(dtype), shape)
            for shape, dtype in shapes
        ])

    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions.extend([1])
    sharding.tile_assignment_devices.extend([0])

    # body
    body = xla_client.XlaBuilder("body")
    in_tuple = ops.Parameter(body, 0, in_tuple_shape)
    counter = ops.GetTupleElement(in_tuple, 0)
    counter = ops.Sub(counter, ops.Constant(body, np.int32(1)))

    operands = [
        ops.GetTupleElement(in_tuple, i + 1) for i in range(len(shapes))
    ]
    body.set_sharding(sharding)
    op_func(operands)
    body.clear_sharding()
    ops.Tuple(body, [counter] + operands)
    body_computation = body.build()

    # condition
    cond = xla_client.XlaBuilder("condition")
    in_tuple = ops.Parameter(cond, 0, in_tuple_shape)
    counter = ops.GetTupleElement(in_tuple, 0)
    ops.Gt(counter, ops.Constant(cond, np.int32(0)))
    cond_computation = cond.Build()

    # while loop
    loop = xla_client.XlaBuilder("loop")
    counter = _op_parameter(loop, 0, (), np.int32)
    operands = [
        _op_parameter(loop, i + 1, shape, dtype)
        for i, (shape, dtype) in enumerate(shapes)
    ]
    while_init = ops.Tuple(loop, [counter] + operands)
    ops.While(cond_computation, body_computation, while_init)
    for i in range(len(shapes) + 1):
        loop.setup_alias((i,), i, ())
    loop_computation = loop.Build()

    compile_options = xla_bridge.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )
    shapes = [(1, np.int32)] + shapes
    return shapes, backend.compile(loop_computation, compile_options)


def profile_hlo_ops(backend, local_devices, num_devices, op_infos):
    results = []
    for op_info in op_infos:
        print(f"Profiling {op_info}")

        if op_info[0] == "matmul":
            n, m, k, dtype = op_info[1]
            shapes = [((n, k), dtype), ((k, m), dtype), ((n, m), dtype)]

            def op_func(operands):
                lhs, rhs, _ = operands
                dim_numbers = (((1,), (0,)), ((), ()))
                dim_numbers = xla_client.make_dot_dimension_numbers(dim_numbers)
                out = ops.DotGeneral(lhs, rhs, dim_numbers)
                operands[-1] = out

            warmup = 2
            number = 10
        elif op_info[0] == "all-reduce":
            replica_groups, dtype, size = op_info[1]
            shapes = [((size,), dtype), ((size,), dtype)]

            def op_func(operands):
                channel_id = backend.create_channel_handle()
                out = _op_all_reduce(operands[0], dtype, "add", replica_groups,
                                     channel_id)
                operands[-1] = out

            warmup = 2
            number = min(
                max(15,
                    int((1 << 31) / (max(size, 1) * np.dtype(dtype).itemsize))),
                1 << 13)
        elif op_info[0] == "reduce-scatter":
            replica_groups, dtype, size = op_info[1]
            shapes = [((size,), dtype),
                      ((size // len(replica_groups[0]),), dtype)]

            def op_func(operands):
                channel_id = backend.create_channel_handle()
                out = _op_reduce_scatter(operands[0], dtype, "add",
                                         replica_groups, channel_id)
                operands[-1] = out

            warmup = 2
            number = min(
                max(15,
                    int((1 << 31) / (max(size, 1) * np.dtype(dtype).itemsize))),
                1 << 13)
        elif op_info[0] == "all-gather":
            replica_groups, dtype, size = op_info[1]
            shapes = [((size // len(replica_groups[0]),), dtype),
                      ((size,), dtype)]

            def op_func(operands):
                if size == 0:
                    return
                channel_id = backend.create_channel_handle()
                out = _op_all_gather(operands[0], replica_groups, channel_id)
                operands[-1] = out

            warmup = 2
            number = min(
                max(15,
                    int((1 << 31) / (max(size, 1) * np.dtype(dtype).itemsize))),
                1 << 13)
        elif op_info[0] == "all-to-all":
            replica_groups, dtype, size = op_info[1]
            shapes = [((size // len(replica_groups[0]),), dtype),
                      ((size // len(replica_groups[0]),), dtype)]

            def op_func(operands):
                if size == 0:
                    return
                channel_id = backend.create_channel_handle()
                out = _op_all_to_all(operands[0], replica_groups, channel_id)
                operands[-1] = out

            warmup = 2
            number = min(
                max(15,
                    int((1 << 31) / (max(size, 1) * np.dtype(dtype).itemsize))),
                1 << 13)
        else:
            raise NotImplementedError(f"Invalid op: {op_info[0]}")

        # Compile
        shapes, compiled = compile_profiling_executable(backend, shapes,
                                                        op_func, num_devices)

        # Warm up
        device_inputs = []
        for i, (shape, dtype) in enumerate(shapes):
            if i == 0:
                device_inputs.append([
                    backend.buffer_from_pyval(np.int32(warmup),
                                              local_devices[i])
                    for i in range(len(local_devices))
                ])
            else:
                device_inputs.append([
                    backend.buffer_from_pyval(np.empty(shape, dtype),
                                              local_devices[i])
                    for i in range(len(local_devices))
                ])
        device_inputs = compiled.execute_sharded_on_local_devices(device_inputs)

        # Run profiling
        device_inputs[0] = \
            [backend.buffer_from_pyval(np.int32(number), local_devices[i])
                for i in range(len(local_devices))]

        [d.synchronize_all_activity() for d in local_devices]
        tic = time.time()
        compiled.execute_sharded_on_local_devices(device_inputs)
        [d.synchronize_all_activity() for d in local_devices]
        toc = time.time()

        results.append((toc - tic) / number)

    return np.array(results)


def profile_matmul(device_cluster):
    physical_mesh = device_cluster.get_physical_mesh(host_ids=[0],
                                                     num_devices_per_host=1)

    # Profile matmul
    op_infos = []
    for dtype in [np.float16, np.float32]:
        for i in range(1, 48):
            n = 128 * i
            op_infos.append(("matmul", (n, n, n, dtype)))
    results = physical_mesh.profile_hlo_ops(op_infos)

    matmul_cost_dict = {}
    matmul_cost_dict[np.float16] = []
    matmul_cost_dict[np.float32] = []
    for i in range(len(op_infos)):
        n, m, k, dtype = op_infos[i][1]
        flop_count = 2 * n * m * k
        matmul_cost_dict[dtype].append((flop_count, flop_count / results[i]))
        print(
            f"Matmul: {(n, m, k, np.dtype(dtype))}, TFLOPS: {flop_count / results[i]/ 1e12:.2f}"
        )

    return matmul_cost_dict


def enumerate_all_collective_spec(num_hosts, num_devices_per_host,
                                  size_configs):
    # Enumerate all possible logical meshes
    logical_mesh_shapes = []
    total_devices = num_hosts * num_devices_per_host
    for i in range(1, total_devices + 1):
        if total_devices % i == 0:
            logical_mesh_shapes.append((total_devices // i, i))

    # Enumerate all replica groups
    all_specs = set()
    for logical_mesh_shape in logical_mesh_shapes:
        # dim 0
        replica_groups = []
        tmp_group = []
        for i in range(logical_mesh_shape[0]):
            tmp_group.append(
                tuple(i * logical_mesh_shape[1] + j
                      for j in range(logical_mesh_shape[1])))
        replica_groups.append(tuple(tmp_group))

        # dim 1
        tmp_group = []
        for j in range(logical_mesh_shape[1]):
            tmp_group.append(
                tuple(i * logical_mesh_shape[1] + j
                      for i in range(logical_mesh_shape[0])))
        replica_groups.append(tuple(tmp_group))

        for replica_group in replica_groups:
            for size, dtype in size_configs:
                all_specs.add((replica_group, dtype, size))
    all_specs = list(all_specs)
    all_specs.sort()
    return all_specs


def profile_all(device_cluster):
    from parax.pipeline_parallel.stage_construction import get_submesh_choices
    print_used_time(None)

    ##### Profile compute cost
    matmul_cost_dict = profile_matmul(device_cluster)
    print_used_time("Profile matmul")

    ##### Profile communication cost

    # Enumerate all size configs
    size_configs = [(1 << 28, "float32")]
    #size_configs = [(0, "float32"), (0, "float16")]
    #for i in range(0, 28):
    #    size_configs.append((1 << i, "float32"))
    #    size_configs.append((1 << i, "float16"))

    virtual_mesh = device_cluster.get_virtual_physical_mesh()
    submesh_choices = get_submesh_choices(virtual_mesh)

    submesh_choices = ((1, 8),)

    for i, (num_hosts, num_devices_per_host) in enumerate(submesh_choices):
        print(f"Mesh {(num_hosts, num_devices_per_host)}")

        # Slice a mesh
        tmp_mesh = virtual_mesh.slice_2d(list(range(num_hosts)),
                                         np.arange(num_hosts * num_devices_per_host).\
                                         reshape((num_hosts, num_devices_per_host)))
        all_specs = enumerate_all_collective_spec(num_hosts,
                                                  num_devices_per_host,
                                                  size_configs)

        op_infos = []
        for op_type in ["all-reduce", "all-gather", "reduce-scatter", "all-to-all"]:
            for spec in all_specs:
                op_infos.append((op_type, spec))

        physical_mesh = tmp_mesh.get_physical_mesh()
        results = physical_mesh.profile_hlo_ops(op_infos)

        all_reduce_cost_dict = defaultdict(list)
        all_gather_cost_dict = defaultdict(list)
        reduce_scatter_cost_dict = defaultdict(list)
        all_to_all_cost_dict = defaultdict(list)

        for i in range(len(op_infos)):
            op_type, (replica_groups, dtype, size) = op_infos[i]
            array_size = size * np.dtype(dtype).itemsize
            num_devices = len(replica_groups[0])

            if op_type == "all-reduce":
                communication_size = 2 * array_size * (num_devices -
                                                       1) / num_devices
            elif op_type == "all-gather" or op_type == "reduce-scatter":
                communication_size = array_size * (num_devices -
                                                   1) / num_devices
            elif op_type == "all-to-all":
                communication_size = array_size * (
                    num_devices - 1) / num_devices / num_devices
            else:
                raise ValueError(f"Invalid op: {op_type}")

            bandwidth = communication_size / results[i]
            print(f"Op: {op_infos[i]}, Bandwidth: {bandwidth / GB} GB/s")

        physical_mesh.shutdown()
    print_used_time("Profile communication")
