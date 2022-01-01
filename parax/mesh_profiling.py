# pylint: disable=no-self-use
"""Profiling communication cost."""
from collections import defaultdict
import os
import pickle
import time
import ray

import numpy as np

from jax.lib import xla_client, xla_bridge, xla_extension

from parax.util import GB, print_used_time, XlaPassContext

ops = xla_client.ops


class MeshProfilingResult:
    """Store the profiling result for a physical mesh."""

    def __init__(self):
        # Cost dictionary for communication primitives.
        # Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
        # The elements in the list is sorted according to the size (ascending).
        self.all_gather_cost_dict = defaultdict(list)
        self.all_reduce_cost_dict = defaultdict(list)
        self.all_to_all_cost_dict = defaultdict(list)
        self.reduce_scatter_cost_dict = defaultdict(list)
        self.available_memory_per_device = None

        # Cost dictionary for computation primitives.
        # Reuse the same data structure.
        # Dict[Tuple(None, dtype)] -> List[Tuple(flop_count, time)]
        self.dot_cost_dict = defaultdict(list)
        self.conv_cost_dict = []

        # Cost dictionary for specific operators
        # Dict[op_info] -> double
        self.op_cost_dict = []

    def update(self, new_mesh_result):
        pass

    def estimate_all_gather(self, group, size, dtype):
        ret = (
            self._estimate_internal(group, size, dtype,
                                    self.all_gather_cost_dict) -
            self._estimate_internal(group, 0, dtype, self.all_gather_cost_dict))
        return ret

    def estimate_all_reduce(self, group, size, dtype):
        ret = (
            self._estimate_internal(group, size, dtype,
                                    self.all_reduce_cost_dict) -
            self._estimate_internal(group, 0, dtype, self.all_reduce_cost_dict))
        return ret

    def multiply_scale(self, factor):
        """
        Multiply the time cost by a constant factor.

        This is used to make the scale of time cost similar to the old alpha-beta model.
        """
        self._multiply_scale_internal(factor, self.all_gather_cost_dict)
        self._multiply_scale_internal(factor, self.all_reduce_cost_dict)
        self._multiply_scale_internal(factor, self.all_to_all_cost_dict)
        self._multiply_scale_internal(factor, self.reduce_scatter_cost_dict)

    def make_monotonic(self):
        """Make the bandwidth monotonically increase along with the communication size."""
        self._make_monotonic_internal(self.all_gather_cost_dict)
        self._make_monotonic_internal(self.all_reduce_cost_dict)
        self._make_monotonic_internal(self.all_to_all_cost_dict)
        self._make_monotonic_internal(self.reduce_scatter_cost_dict)
        self._make_monotonic_internal(self.dot_cost_dict)

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
            comm_bytes = 2 * (num_devices -
                              1) / num_devices * sizes * to_np_dtype(
                                  key[1]).itemsize
            bandwidth = comm_bytes / times / GB
            bandwidth_str = ", ".join(f"{x:.2f}" for x in bandwidth)
            ret += f"Key: {key}\nBandwidth: {bandwidth_str}\n"
        return ret


class ProfilingResultDatabase:

    def __init__(self, data={}):
        self.data = data

    def query(self, cluster_key, mesh_shape):
        key = (cluster_key, mesh_shape)
        return self.data[key]

    def update_one_mesh(self, cluster_key, mesh_shape, mesh_result):
        key = (cluster_key, mesh_shape)
        if key not in self.data:
            self.data[key] = mesh_result
        else:
            self.data[key].update(mesh_result)

    def update(self, new_database):
        for ((cluster_key, mesh_shape),
             mesh_result) in new_database.data.items():
            self.update_one_mesh(cluster_key, mesh_shape, mesh_result)

    def save(self, filename):
        pickle.dump(self.data, open(filename, "wb"))

    def load(self, filename):
        new_data = pickle.load(open(filename, "rb"))
        self.update(ProfilingResultDatabase(new_data))

    def __str__(self):
        ret = ""
        for (cluster_key, mesh_shape), value in self.data.items():
            ret += f"cluster_key: {cluster_key}, mesh_shape: {mesh_shape}\n"
            ret += str(value)
        return ret


def _op_parameter(builder, num, shape, dtype):
    shape = xla_client.Shape.array_shape(dtype, shape)
    name = ""
    replicated = []
    return ops.Parameter(builder, num,
                         shape.with_major_to_minor_layout_if_absent(), name,
                         replicated)


def _op_all_gather(operand, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    ret = ops.AllGather(operand, 0, len(replica_groups[0]),
                        replica_groups_protos, channel_id, None, True)
    return ret


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


def _op_all_to_all(operand, replica_groups, channel_id):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    ret = ops.AllToAll(operand, 0, 0, len(replica_groups[0]),
                       replica_groups_protos, channel_id, None, True)
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


def _compile_profiling_executable(backend, shapes, op_func, num_devices):
    """
    Compile a xla executable for benchmarking operators.
    It is a while loop that calls the operator for multiple times.
    """

    in_tuple_shape = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(np.dtype(np.int32), ())] +
        [xla_client.Shape.array_shape(dtype, shape) for shape, dtype in shapes])

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
    counter = _op_parameter(loop, 0, (), np.dtype(np.int32))
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


def to_np_dtype(dtype_str):
    if dtype_str == "f32":
        return np.dtype("float32")
    elif dtype_str == "f16":
        return np.dtype("float16")
    else:
        return np.dtype(dtype_str)


def rank_0_print(host_id, msg):
    if host_id == 0:
        print(msg, flush=True)


def profile_one_hlo_op(backend, local_devices, host_id, num_devices,
                       num_devices_per_node, op_info):
    if op_info[0] == "dot":
        n, m, k, dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        shapes = [((n, k), dtype), ((k, m), dtype), ((n, m), dtype)]

        def op_func(operands):
            lhs, rhs, _ = operands
            dim_numbers = (((1,), (0,)), ((), ()))
            dim_numbers = xla_client.make_dot_dimension_numbers(dim_numbers)
            out = ops.DotGeneral(lhs, rhs, dim_numbers)
            operands[-1] = out

        flop_ct = max(2 * n * m * k, 1)
        if dtype_str == "f16":
            work = 50e12
        elif dtype_str == "f32":
            work = 10e12
        else:
            raise ValueError(f"Invalid type: {dtype_str}")
        number = min(max(12, int(work / flop_ct)), 1 << 12)
    elif op_info[0] == "all-gather":
        replica_groups, dtype, size = op_info[1]
        dtype = to_np_dtype(dtype)
        size = size // len(replica_groups[0]) * len(replica_groups[0])
        shapes = [((size // len(replica_groups[0]),), dtype), ((size,), dtype)]

        def op_func(operands):
            if shapes[0][0][0] == 0:
                return
            channel_id = backend.create_channel_handle()
            out = _op_all_gather(operands[0], replica_groups, channel_id)
            operands[-1] = out

        if max(replica_groups[0]) - min(
                replica_groups[0]) < num_devices_per_node:
            work = 1 << 33
        else:
            work = 1 << 31
        number = min(max(12, int(work / max(size * dtype.itemsize, 1))),
                     1 << 13)
    elif op_info[0] == "all-reduce":
        replica_groups, dtype, size = op_info[1]
        dtype = to_np_dtype(dtype)
        shapes = [((size,), dtype), ((size,), dtype)]

        def op_func(operands):
            channel_id = backend.create_channel_handle()
            out = _op_all_reduce(operands[0], dtype, "add", replica_groups,
                                 channel_id)
            operands[-1] = out

        if max(replica_groups[0]) - min(
                replica_groups[0]) < num_devices_per_node:
            work = 1 << 32
        else:
            work = 1 << 30
        number = min(max(12, int(work / max(size * dtype.itemsize, 1))),
                     1 << 13)
    elif op_info[0] == "all-to-all":
        replica_groups, dtype, size = op_info[1]
        dtype = to_np_dtype(dtype)
        size = size // (len(replica_groups[0])**2) * (len(replica_groups[0])**2)
        shapes = [((size // len(replica_groups[0]),), dtype),
                  ((size // len(replica_groups[0]),), dtype)]

        def op_func(operands):
            if shapes[0][0][0] // len(replica_groups[0]) == 0:
                return
            channel_id = backend.create_channel_handle()
            out = _op_all_to_all(operands[0], replica_groups, channel_id)
            operands[-1] = out

        if max(replica_groups[0]) - min(
                replica_groups[0]) < num_devices_per_node:
            work = 1 << 33
        else:
            work = 1 << 31
        number = min(max(12, int(work / max(size * dtype.itemsize, 1))),
                     1 << 13)
    elif op_info[0] == "reduce-scatter":
        replica_groups, dtype, size = op_info[1]
        dtype = to_np_dtype(dtype)
        size = size // len(replica_groups[0]) * len(replica_groups[0])
        shapes = [((size,), dtype), ((size // len(replica_groups[0]),), dtype)]

        def op_func(operands):
            if shapes[1][0][0] == 0:
                return
            channel_id = backend.create_channel_handle()
            out = _op_reduce_scatter(operands[0], dtype, "add", replica_groups,
                                     channel_id)
            operands[-1] = out

        if max(replica_groups[0]) - min(
                replica_groups[0]) < num_devices_per_node:
            work = 1 << 33
        else:
            work = 1 << 31
        number = min(max(12, int(work / max(size * dtype.itemsize, 1))),
                     1 << 13)
    else:
        raise NotImplementedError(f"Invalid op: {op_info[0]}")

    warmup = max(number // 10, 2)
    rank_0_print(
        host_id, f"Profiling {op_info}, work: {work}, number: {number}, "
        f"time: {time.time():.0f}.")

    # Compile
    shapes, compiled = _compile_profiling_executable(backend, shapes, op_func,
                                                     num_devices)

    # Warm up
    device_inputs = []
    for j, (shape, dtype) in enumerate(shapes):
        if j == 0:
            device_inputs.append([
                backend.buffer_from_pyval(np.int32(warmup), local_devices[k])
                for k in range(len(local_devices))
            ])
        else:
            device_inputs.append([
                backend.buffer_from_pyval(np.empty(shape, dtype),
                                          local_devices[k])
                for k in range(len(local_devices))
            ])

    [d.synchronize_all_activity() for d in local_devices]
    device_inputs = compiled.execute_sharded_on_local_devices(device_inputs)

    # Run profiling
    device_inputs[0] = [
        backend.buffer_from_pyval(np.int32(number), local_devices[k])
        for k in range(len(local_devices))
    ]

    [d.synchronize_all_activity() for d in local_devices]
    tic = time.time()
    compiled.execute_sharded_on_local_devices(device_inputs)
    [d.synchronize_all_activity() for d in local_devices]
    toc = time.time()

    mean_time = (toc - tic) / number
    return mean_time


def profile_hlo_ops(op_infos, backend, local_devices, host_id, num_devices,
                    cache_filename):
    results = []
    num_devices_per_node = 8
    save_every = 15

    if os.path.exists(cache_filename):
        rank_0_print(host_id,
                     f"Load cached hlo op cost dict from {cache_filename}...")
        cache_dict = pickle.load(open(cache_filename, "rb"))
    else:
        cache_dict = {}

    for i, op_info in enumerate(op_infos):
        if op_info in cache_dict:
            rank_0_print(host_id, f"Hit cache {op_info}...")
            results.append(cache_dict[op_info])
            continue

        mean_time = profile_one_hlo_op(backend, local_devices, host_id,
                                       num_devices, num_devices_per_node,
                                       op_info)
        cache_dict[op_info] = mean_time
        results.append(mean_time)

        if host_id == 0 and ((i + 1) % save_every == 0 or
                             i == len(op_infos) - 1):
            rank_0_print(host_id, "Save cache...")
            pickle.dump(cache_dict, open(cache_filename, "wb"))

    return np.array(results)


def profile_dot(device_cluster, cache_filename):
    physical_mesh = device_cluster.get_physical_mesh(host_ids=[0],
                                                     num_devices_per_host=1)

    # Profile dot
    op_infos = []
    for dtype in ["f16", "f32"]:
        for i in range(0, 48):
            n = 128 * i
            op_infos.append(("dot", (n, n, n, dtype)))
    results = physical_mesh.profile_hlo_ops(op_infos, cache_filename)

    dot_cost_dict = defaultdict(list)
    for i in range(len(op_infos)):
        n, m, k, dtype = op_infos[i][1]
        flop_count = 2 * n * m * k
        dot_cost_dict[((), dtype)].append((flop_count, results[i]))
        print(
            f"Matmul: {(n, m, k, dtype)}, TFLOPS: {flop_count / results[i]/ 1e12:.2f}"
        )

    return dot_cost_dict


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
                all_specs.add((tuple(replica_group), dtype, size))
    all_specs = list(all_specs)
    all_specs.sort()
    return list(all_specs)


def profile_all(device_cluster, cluster_key, comm_size_range, cache_filename):
    """Profile costs for all dot and comuniation primitives."""
    from parax.pipeline_parallel.stage_construction import get_submesh_choices
    print_used_time(None)

    ##### Profile compute cost
    dot_cost_dict = profile_dot(device_cluster, cache_filename)
    print_used_time("Profile dot")

    ##### Profile communication cost

    # Enumerate all size configs
    size_configs = [(0, "f32"), (0, "f16")]
    for i in range(*comm_size_range):
        size_configs.append((1 << i, "f32"))
        size_configs.append((1 << i, "f16"))

    virtual_mesh = device_cluster.get_virtual_physical_mesh()
    submesh_choices = list(reversed(get_submesh_choices(virtual_mesh)))

    prof_database = ProfilingResultDatabase()
    for i, (num_hosts, num_devices_per_host) in enumerate(submesh_choices):
        print(f"Mesh shape: {(num_hosts, num_devices_per_host)}")

        # Slice a mesh
        tmp_mesh = virtual_mesh.slice_2d(
            list(range(num_hosts)),
            np.arange(num_hosts * num_devices_per_host).reshape(
                (num_hosts, num_devices_per_host)))
        all_specs = enumerate_all_collective_spec(num_hosts,
                                                  num_devices_per_host,
                                                  size_configs)

        op_infos = []
        for op_type in [
                "all-gather", "all-reduce", "all-to-all", "reduce-scatter"
        ]:
            for spec in all_specs:
                op_infos.append((op_type, spec))

        physical_mesh = tmp_mesh.get_physical_mesh()
        available_memory_per_device = physical_mesh.get_available_memory()

        # Profile operators in batch to resolve some deadlock issues
        batch_size = 30
        batch_timeout = batch_size * 20
        results = []
        s = 0
        while s < len(op_infos):
            try:
                batch_result = physical_mesh.profile_hlo_ops(
                    op_infos[s:s + batch_size],
                    cache_filename,
                    timeout=batch_timeout)
            except ray.exceptions.RayError:
                physical_mesh.shutdown(forced=True)
                physical_mesh = None
                time.sleep(30)
                physical_mesh = tmp_mesh.get_physical_mesh()
                continue
            #batch_result = physical_mesh.profile_hlo_ops(op_infos[s:s + batch_size],
            #                                             timeout=batch_timeout)
            results.extend(batch_result)
            s += batch_size

        # Parse results
        all_gather_cost_dict = defaultdict(list)
        all_reduce_cost_dict = defaultdict(list)
        all_to_all_cost_dict = defaultdict(list)
        reduce_scatter_cost_dict = defaultdict(list)
        for i in range(len(op_infos)):
            op_type, (replica_groups, dtype, size) = op_infos[i]
            array_size = size * to_np_dtype(dtype).itemsize
            num_devices = len(replica_groups[0])

            if op_type == "all-gather":
                communication_size = array_size * (num_devices -
                                                   1) / num_devices
                all_gather_cost_dict[(replica_groups, dtype)].append(
                    (size, results[i]))
            elif op_type == "all-reduce":
                communication_size = 2 * array_size * (num_devices -
                                                       1) / num_devices
                all_reduce_cost_dict[(replica_groups, dtype)].append(
                    (size, results[i]))
            elif op_type == "all-to-all":
                communication_size = array_size * (
                    num_devices - 1) / num_devices / num_devices
                all_to_all_cost_dict[(replica_groups, dtype)].append(
                    (size, results[i]))
            elif op_type == "reduce-scatter":
                communication_size = array_size * (num_devices -
                                                   1) / num_devices
                reduce_scatter_cost_dict[(replica_groups, dtype)].append(
                    (size, results[i]))
            else:
                raise ValueError(f"Invalid op: {op_type}")

            bandwidth = communication_size / results[i]
            print(f"Op: {op_infos[i]}, Bandwidth: {bandwidth / GB:.2f} GB/s")

        physical_mesh.shutdown()

        mesh_result = MeshProfilingResult()
        mesh_result.dot_cost_dict = dot_cost_dict
        mesh_result.all_gather_cost_dict = all_gather_cost_dict
        mesh_result.all_reduce_cost_dict = all_reduce_cost_dict
        mesh_result.all_to_all_cost_dict = all_to_all_cost_dict
        mesh_result.reduce_scatter_cost_dict = reduce_scatter_cost_dict
        mesh_result.available_memory_per_device = available_memory_per_device
        mesh_result.make_monotonic()
        prof_database.update_one_mesh(cluster_key,
                                      (num_hosts, num_devices_per_host),
                                      mesh_result)

    print_used_time("Profile communication")
    return prof_database


def estimate_hlo_module_cost(hlo_module,
                             profiling_results,
                             num_micro_batches=1,
                             grad_sync_channel_ids=""):
    with XlaPassContext({
            "gpu_cost_model::profiling_results": profiling_results,
            "gpu_cost_model::num_micro_batches": num_micro_batches,
            "gpu_cost_model::grad_sync_channel_ids": grad_sync_channel_ids,
            "gpu_cost_model::verbose": 0,
    }):
        return xla_extension.estimate_hlo_module_cost(hlo_module)
