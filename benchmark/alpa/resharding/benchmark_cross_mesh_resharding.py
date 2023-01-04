"""Test cross-mesh resharding."""
import argparse

from jax import xla
from jax.core import Var
from jax._src.abstract_arrays import ShapedArray
from jax.interpreters.pxla import spec_to_indices
import jax.numpy as jnp
import numpy as np
import ray

from alpa import init
from alpa.device_mesh import (create_remote_array_refs,
                              get_global_virtual_physical_mesh)
from alpa.mesh_executable import next_mesh_executable_uuid
from alpa.global_env import global_config
from alpa.pipeline_parallel.runtime_emitter import PipelineInstEmitter
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CollectiveGroup, ReshardingTaskSpec, CrossMeshCommunicator,
    SymbolicReshardingTask, SymbolicBroadcastReshardingTask)
from alpa.pipeline_parallel.pipeshard_executable import (
    AllocateZeroWorkerExecutableConfig, PipelineInstruction,
    PipeshardMeshWorkerExecutable)
from alpa.pipeline_parallel.resharding_tensor import VirtualDistributedArray
from alpa.util import get_shard_shape
from alpa.timer import timers

import suite


def get_device_meshes(src_mesh_shape, dst_mesh_shape):
    virtual_mesh = get_global_virtual_physical_mesh()
    src_num_host = src_mesh_shape[0]
    dst_num_host = dst_mesh_shape[0]
    assert virtual_mesh.num_hosts >= src_num_host+dst_num_host,\
        "Error: There are not enough nodes for this test case"
    src_mesh = virtual_mesh.slice_2d(range(src_num_host),
                                     [range(src_mesh_shape[1])] *
                                     src_num_host).get_physical_mesh()
    dst_host_indices = range(src_num_host, src_num_host + dst_num_host)
    dst_device_indices = [range(dst_mesh_shape[1])] * dst_num_host
    dst_mesh = virtual_mesh.slice_2d(dst_host_indices,
                                     dst_device_indices).get_physical_mesh()
    return src_mesh, dst_mesh


def get_mean_and_variance(results):
    assert len(results) == 13
    results = results[3:]
    mean = np.mean(results)
    var = np.var(results)
    return mean, var


def benchmark_one_case_internal(
    src_mesh_shape,
    dst_mesh_shape,
    src_sharding_spec,
    dst_sharding_spec,
    tensor_shape,
    resharding_mode="send_recv",
    use_local_allgather=True,
    resharding_loadbalance_mode="normal",
):

    global_config.resharding_mode = resharding_mode
    global_config.resharding_loadbalance_mode = resharding_loadbalance_mode
    global_config.use_local_allgather = use_local_allgather

    init(cluster="ray")

    src_mesh, dst_mesh = get_device_meshes(src_mesh_shape, dst_mesh_shape)

    var = Var(0, "", ShapedArray(tensor_shape, jnp.int32))

    # Resharding task spec and send/recv strategy
    src_loads = {src: 0 for src in src_mesh.device_strs}
    dst_loads = {dst: 0 for dst in dst_mesh.device_strs}
    if resharding_mode == "send_recv":
        rewrite_dst_sharding_spec = CrossMeshCommunicator._rewrite_allgather_spec(
            dst_sharding_spec, dst_mesh.num_hosts, var.aval.shape)
    else:
        rewrite_dst_sharding_spec = dst_sharding_spec
    src_array = VirtualDistributedArray(device_mesh=src_mesh,
                                        aval=var.aval,
                                        sharding_spec=src_sharding_spec)
    dst_array = VirtualDistributedArray(device_mesh=dst_mesh,
                                        aval=var.aval,
                                        sharding_spec=rewrite_dst_sharding_spec)
    task_spec = ReshardingTaskSpec(src_array, dst_array, dst_sharding_spec)

    if resharding_mode == "send_recv":
        if global_config.resharding_loadbalance_mode == "normal":
            strategy = (CrossMeshCommunicator.
                        _generate_send_recv_resharding_strategy_by_loads(
                            task_spec, src_loads, dst_loads))
        elif global_config.resharding_loadbalance_mode == "no_loadbalance":
            strategy = (
                CrossMeshCommunicator.
                _generate_send_recv_resharding_strategy_by_no_load(task_spec))
        elif global_config.resharding_loadbalance_mode in [
                "loadbalance_size", "loadbalance_order"
        ]:
            strategy = (CrossMeshCommunicator.
                        _generate_send_recv_resharding_strategy_by_loadbalance(
                            task_spec, src_mesh, dst_mesh))
    else:
        if global_config.resharding_loadbalance_mode == "normal":
            strategy = (CrossMeshCommunicator.
                        _generate_broadcast_resharding_strategy_by_loads(
                            task_spec, src_loads, dst_loads))
        elif global_config.resharding_loadbalance_mode == "no_loadbalance":
            strategy = (
                CrossMeshCommunicator.
                _generate_broadcast_resharding_strategy_by_no_load(task_spec))
        elif global_config.resharding_loadbalance_mode in [
                "loadbalance_size", "loadbalance_order"
        ]:
            strategy = (CrossMeshCommunicator.
                        _generate_broadcast_resharding_strategy_by_loadbalance(
                            task_spec, src_mesh, dst_mesh))

    task_spec.set_resharding_strategy(strategy)

    # Resharding task. Compile send/recv from strategy and allgather.
    collective_group = CollectiveGroup(task_spec.get_participant_device_strs(),
                                       src_mesh, dst_mesh)
    if global_config.eagerly_create_communicators:
        collective_group.instantiate_now()
    else:
        collective_group.instantiate()
    if resharding_mode == "send_recv":
        task = SymbolicReshardingTask(task_spec, collective_group, src_mesh,
                                      dst_mesh)
    else:
        task = SymbolicBroadcastReshardingTask(task_spec, collective_group,
                                               src_mesh, dst_mesh)

    if global_config.eagerly_create_communicators:
        task.create_resharding_communicators()

    # Compile pipeline instructions
    instruction_lists = {worker: [] for worker in src_mesh.workers}
    for worker in dst_mesh.workers:
        instruction_lists[worker] = []
    executable_config_lists = {worker: [] for worker in dst_mesh.workers}
    src_uuid = 21474
    dst_uuid = 21475
    # allocate the buffer
    exec_uuid = next_mesh_executable_uuid()
    config = AllocateZeroWorkerExecutableConfig(
        exec_uuid, [get_shard_shape(var.aval, rewrite_dst_sharding_spec)],
        [var.aval.dtype])
    output_uuids = [dst_uuid]
    for worker in dst_mesh.workers:
        executable_config_lists[worker].append(config)
        in_uuids = []
        out_uuids = output_uuids
        instruction_lists[worker].append(
            PipelineInstruction.run(config.exec_uuid,
                                    in_uuids,
                                    out_uuids, {
                                        "sync_before": False,
                                        "sync_after": False
                                    },
                                    info="allocate zero for recv"))
    # Create resharding task

    if resharding_mode == "send_recv":
        PipelineInstEmitter._compile_resharding_task(src_uuid, task, dst_uuid,
                                                     instruction_lists)
    else:
        PipelineInstEmitter._compile_broadcast_resharding_task(
            src_mesh, src_uuid, task, dst_uuid, instruction_lists)

    exec_uuids = {}

    # Compile Pipeline Executable
    for worker in src_mesh.workers:
        exec_uuid = next_mesh_executable_uuid()
        # print(worker, exec_uuid)
        worker.put_executable.remote(exec_uuid, PipeshardMeshWorkerExecutable,
                                     instruction_lists[worker], [src_uuid], [],
                                     [], [], [],
                                     [False] * src_mesh.num_devices_per_host)
        exec_uuids[worker] = exec_uuid
    for worker in dst_mesh.workers:
        exec_uuid = next_mesh_executable_uuid()
        # print(worker, exec_uuid)
        worker.put_executable.remote(exec_uuid, PipeshardMeshWorkerExecutable,
                                     instruction_lists[worker], [], [dst_uuid],
                                     executable_config_lists[worker], [], [],
                                     [False] * dst_mesh.num_devices_per_host)
        exec_uuids[worker] = exec_uuid

    # Prepare array and shard args
    test_array = np.arange(np.prod(var.aval.shape),
                           dtype=var.aval.dtype).reshape(var.aval.shape)
    indices = spec_to_indices(var.aval.shape, src_sharding_spec)
    test_array = xla.canonicalize_dtype(test_array)
    input_refs = src_mesh.shard_args_to_bufs([indices], (False,), (False,),
                                             None, [test_array])
    input_refs = np.array(input_refs)
    input_uuids = [ref.uuid for ref in input_refs]
    output_refs, output_uuids = create_remote_array_refs(dst_mesh)

    # Run executables
    time_spend = []
    for _ in range(13):
        timers("overall_resharding_time").start()
        for worker in src_mesh.workers:
            worker.run_executable.remote(exec_uuids[worker],
                                         input_uuids, [],
                                         sync_for_timer=True,
                                         collect_trace=False)
        for worker in dst_mesh.workers:
            worker.run_executable.remote(exec_uuids[worker], [],
                                         output_uuids,
                                         sync_for_timer=True,
                                         collect_trace=False)

        dst_mesh.sync_workers(sync_all_devices=True)
        timers("overall_resharding_time").stop()
        time_spend.append(timers("overall_resharding_time").elapsed(mode="sum"))
        timers("overall_resharding_time").reset()

    mean_time, var_time = get_mean_and_variance(time_spend)
    result = {
        "src_mesh_shape": src_mesh_shape,
        "dst_mesh_shape": dst_mesh_shape,
        "src_sharding_spec": str(src_sharding_spec),
        "dst_sharding_spec": str(dst_sharding_spec),
        "tensor_shape": tensor_shape,
        "resharding_mode": resharding_mode,
        "use_local_allgather": use_local_allgather,
        "resharding_loadbalance_mode": resharding_loadbalance_mode,
        "exec_time_mean": mean_time,
        "exec_time_var": var_time
    }

    # Delete executables
    for worker in src_mesh.workers:
        worker.delete_executable.remote(exec_uuids[worker])
    for worker in dst_mesh.workers:
        worker.delete_executable.remote(exec_uuids[worker])

    src_mesh.shutdown()
    dst_mesh.shutdown()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        type=str,
                        required=True,
                        choices=["1-to-m", "n-to-m"])
    parser.add_argument("--case", type=str)
    parser.add_argument("--n-nodes", type=int, default=1)
    parser.add_argument("--gpu-per-node", type=int, default=1)
    parser.add_argument("--resharding-mode",
                        type=str,
                        required=True,
                        choices=["send_recv", "broadcast"])
    parser.add_argument("--resharding-loadbalance-mode",
                        type=str,
                        required=True,
                        choices=[
                            "normal", "no_loadbalance", "loadbalance_size",
                            "loadbalance_order"
                        ])
    parser.add_argument("--use-local-allgather", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    if args.suite == "1-to-m":
        case = suite.perf_1_to_m_suite[(args.n_nodes, args.gpu_per_node)]
    else:
        case = suite.perf_n_to_m_suite[args.case]

    result = benchmark_one_case_internal(
        case.src_mesh_shape, case.dst_mesh_shape, case.src_sharding_spec,
        case.dst_sharding_spec, case.tensor_shape, args.resharding_mode,
        args.use_local_allgather, args.resharding_loadbalance_mode)
    print(result)

# python benchmark_cross_mesh_resharding.py --case case1 --resharding-mode broadcast --resharding-loadbalance-mode normal
