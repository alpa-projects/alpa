"""Test cross-mesh resharding."""
import unittest
from alpa.pipeline_parallel.runtime_emitter import PipelineInstEmitter

from jax import xla
from jax.core import Var
from jax._src.abstract_arrays import ShapedArray
from jax.interpreters.pxla import (Chunked, NoSharding, Replicated, ShardedAxis,
                                   ShardingSpec, spec_to_indices)
import jax.numpy as jnp
import numpy as np

from alpa import init
from alpa.device_mesh import (DistributedArray,
                              get_global_virtual_physical_mesh)
from alpa.mesh_executable import (create_remote_buffer_refs, get_uuid_np_array,
                                  next_mesh_executable_uuid)
from alpa.global_env import global_config
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CollectiveGroup, ReshardingTaskSpec, CrossMeshCommunicator,
    SymbolicReshardingTask, SymbolicBroadcastReshardingTask)
from alpa.pipeline_parallel.pipeshard_executable import (
    AllocateZeroWorkerExecutableConfig, PipelineInstruction,
    PipeshardMeshWorkerExecuable)
from alpa.pipeline_parallel.resharding_tensor import VirtualDistributedArray
from alpa.testing import assert_allclose
from alpa.util import get_shard_shape


def test_resharding(var,
                    src_mesh,
                    src_sharding_spec,
                    dst_mesh,
                    dst_sharding_spec,
                    use_scatter_gather,
                    resharding_mode,
                    src_loads=None,
                    dst_loads=None):
    global_config.use_scatter_gather = use_scatter_gather
    global_config.resharding_mode = resharding_mode

    # Resharding task spec and send/recv strategy
    src_loads = src_loads or {src: 0 for src in src_mesh.device_strs}
    dst_loads = dst_loads or {dst: 0 for dst in dst_mesh.device_strs}
    if resharding_mode == "send_recv":
        (rewrite_dst_sharding_spec,
         local_chunks) = CrossMeshCommunicator._rewrite_allgather_spec(
             dst_sharding_spec, dst_mesh, var.aval.shape)
    else:
        rewrite_dst_sharding_spec = dst_sharding_spec
        local_chunks = None
    src_array = VirtualDistributedArray(device_mesh=src_mesh,
                                        aval=var.aval,
                                        sharding_spec=src_sharding_spec)
    dst_array = VirtualDistributedArray(device_mesh=dst_mesh,
                                        aval=var.aval,
                                        sharding_spec=rewrite_dst_sharding_spec)
    task_spec = ReshardingTaskSpec(src_array, dst_array, local_chunks)
    if resharding_mode == "send_recv":
        strategy = CrossMeshCommunicator._generate_send_recv_resharding_strategy_by_loads(
            task_spec, src_loads, dst_loads)
    else:
        strategy = CrossMeshCommunicator._generate_broadcast_resharding_strategy_by_loads(
            task_spec, src_loads, dst_loads)
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

    # Compile pipeline instructions
    instruction_lists = {worker: [] for worker in src_mesh.workers}
    for worker in dst_mesh.workers:
        instruction_lists[worker] = []
    executable_config_lists = {worker: [] for worker in dst_mesh.workers}
    src_uuids = np.arange(np.prod(src_mesh.shape)).reshape(src_mesh.shape)
    dst_uuids = np.arange(np.prod(dst_mesh.shape)).reshape(dst_mesh.shape)
    # allocate the buffer
    exec_uuid = next_mesh_executable_uuid()
    config = AllocateZeroWorkerExecutableConfig(
        exec_uuid, [get_shard_shape(var.aval, dst_sharding_spec)],
        [var.aval.dtype])
    output_uuids = np.expand_dims(dst_uuids, axis=1)
    for worker_idx, worker in enumerate(dst_mesh.workers):
        executable_config_lists[worker].append(config)
        in_uuids = []
        out_uuids = output_uuids[worker_idx]
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
        PipelineInstEmitter._compile_resharding_task(src_mesh, dst_mesh,
                                                     src_uuids, task, dst_uuids,
                                                     instruction_lists)
    else:
        PipelineInstEmitter._compile_broadcast_resharding_task(
            src_mesh, dst_mesh, src_uuids, task, dst_uuids, instruction_lists)

    exec_uuids = {}

    # Compile Pipeline Executable
    for worker_idx, worker in enumerate(src_mesh.workers):
        exec_uuid = next_mesh_executable_uuid()
        worker.put_executable.remote(exec_uuid, PipeshardMeshWorkerExecuable,
                                     instruction_lists[worker],
                                     [src_uuids[worker_idx]], [], [], [], [],
                                     [False] * src_mesh.num_devices_per_host)
        exec_uuids[worker] = exec_uuid
    for worker_idx, worker in enumerate(dst_mesh.workers):
        exec_uuid = next_mesh_executable_uuid()
        worker.put_executable.remote(exec_uuid, PipeshardMeshWorkerExecuable,
                                     instruction_lists[worker], [],
                                     [dst_uuids[worker_idx]],
                                     executable_config_lists[worker], [], [],
                                     [False] * dst_mesh.num_devices_per_host)
        exec_uuids[worker] = exec_uuid

    # Prepare array and shard args
    test_array = np.arange(np.prod(var.aval.shape),
                           dtype=var.aval.dtype).reshape(var.aval.shape)
    indices = spec_to_indices(var.aval.shape, src_sharding_spec)
    test_array = xla.canonicalize_dtype(test_array)
    input_refs = src_mesh.shard_args_to_bufs([indices], (False,), (False,),
                                             None, [test_array])[0]
    input_refs = np.array(input_refs).reshape(src_mesh.shape)
    input_uuids = get_uuid_np_array(input_refs)
    output_refs, output_uuids = create_remote_buffer_refs(dst_mesh)
    output_uuids = output_uuids.reshape(dst_mesh.shape)

    # Run executables
    # for _ in range(3):
    # timers("overall_resharding_time").start()
    for worker_idx, worker in enumerate(src_mesh.workers):
        worker.run_executable.remote(exec_uuids[worker],
                                     [input_uuids[worker_idx]], [],
                                     sync_for_timer=True)
    for worker_idx, worker in enumerate(dst_mesh.workers):
        worker.run_executable.remote(exec_uuids[worker], [],
                                     [output_uuids[worker_idx]],
                                     sync_for_timer=True)
    output_array = DistributedArray(dst_mesh, var.aval, dst_sharding_spec,
                                    output_refs)

    # dst_mesh.sync_workers()
    # timers("overall_resharding_time").stop()
    # timers("overall_resharding_time").log()
    # timers("overall_resharding_time").reset()

    # Check correctness
    assert_allclose(test_array, output_array)

    # Delete executables
    for worker in src_mesh.workers:
        worker.delete_executable.remote(exec_uuids[worker])
    for worker in dst_mesh.workers:
        worker.delete_executable.remote(exec_uuids[worker])


class ReshardingTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def run_resharding_task(self,
                            src_mesh_shape,
                            dst_mesh_shape,
                            src_sharding_spec,
                            dst_sharding_spec,
                            tensor_shape,
                            use_scatter_gather=True,
                            resharding_mode="send_recv",
                            tensor_dtype=None):
        virtual_mesh = get_global_virtual_physical_mesh()
        src_num_host = src_mesh_shape[0]
        dst_num_host = dst_mesh_shape[0]
        src_mesh = virtual_mesh.slice_2d(range(src_num_host),
                                         [range(src_mesh_shape[1])] *
                                         src_num_host).get_physical_mesh()
        if (src_mesh_shape[1] + dst_mesh_shape[1] <=
                virtual_mesh.num_devices_per_host):
            dst_host_indices = range(dst_num_host)
            dst_device_indices = [
                range(src_mesh_shape[1], src_mesh_shape[1] + dst_mesh_shape[1])
            ] * dst_num_host
        else:
            dst_host_indices = range(src_num_host, src_num_host + dst_num_host)
            dst_device_indices = [range(dst_mesh_shape[1])] * dst_num_host
        dst_mesh = virtual_mesh.slice_2d(
            dst_host_indices, dst_device_indices).get_physical_mesh()

        tensor_dtype = tensor_dtype or jnp.int32
        var = Var(0, "", ShapedArray(tensor_shape, tensor_dtype))
        test_resharding(var, src_mesh, src_sharding_spec, dst_mesh,
                        dst_sharding_spec, use_scatter_gather, resharding_mode)
        src_mesh.shutdown()
        dst_mesh.shutdown()

    def test_4gpu_send_recv(self):
        src_shape = (1, 2)
        dst_shape = (1, 2)
        tensor_shape = (4, 8, 16)
        src_spec = ShardingSpec(
            [NoSharding(), NoSharding(),
             NoSharding()], [Replicated(2)])
        dst_spec = ShardingSpec([Chunked(
            [2]), NoSharding(), NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape)
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape, False)
        src_spec = ShardingSpec([Chunked(
            [2]), NoSharding(), NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape)
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape, False)
        src_spec = ShardingSpec(
            [NoSharding(), Chunked([2]),
             NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape)
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape, False)

    def test_4gpu_allgather(self):
        src_shape = (1, 2)
        dst_shape = (1, 2)
        tensor_shape = (4, 8, 16)
        src_spec = ShardingSpec(
            [NoSharding(), NoSharding(),
             NoSharding()], [Replicated(2)])
        dst_spec = ShardingSpec(
            [NoSharding(), NoSharding(),
             NoSharding()], [Replicated(2)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape)
        src_spec = ShardingSpec([Chunked(
            [2]), NoSharding(), NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape)
        src_spec = ShardingSpec(
            [NoSharding(), Chunked([2]),
             NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape)

    def test_8gpu_2_dim_allgather(self):
        src_shape = (1, 4)
        dst_shape = (1, 4)
        tensor_shape = (2, 8, 16)
        src_spec = ShardingSpec(
            [NoSharding(), NoSharding(),
             NoSharding()], [Replicated(4)])
        dst_spec = ShardingSpec(
            [NoSharding(), NoSharding(),
             NoSharding()], [Replicated(4)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape)

    def test_4gpu_broadcast(self):
        src_shape = (1, 2)
        dst_shape = (1, 2)
        tensor_shape = (4, 8, 16)
        src_spec = ShardingSpec(
            [NoSharding(), NoSharding(),
             NoSharding()], [Replicated(2)])
        dst_spec = ShardingSpec([Chunked(
            [2]), NoSharding(), NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape,
                                 dst_shape,
                                 src_spec,
                                 dst_spec,
                                 tensor_shape,
                                 resharding_mode="broadcast")
        src_spec = ShardingSpec([Chunked(
            [2]), NoSharding(), NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape,
                                 dst_shape,
                                 src_spec,
                                 dst_spec,
                                 tensor_shape,
                                 resharding_mode="broadcast")
        src_spec = ShardingSpec(
            [NoSharding(), Chunked([2]),
             NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape,
                                 dst_shape,
                                 src_spec,
                                 dst_spec,
                                 tensor_shape,
                                 resharding_mode="broadcast")

    def test_8gpu_broadcast(self):
        src_shape = (1, 4)
        dst_shape = (1, 4)
        tensor_shape = (2, 64, 64)

        src_spec = ShardingSpec([Chunked(
            [2]), Chunked([2]), NoSharding()],
                                [ShardedAxis(0), ShardedAxis(1)])
        dst_spec = ShardingSpec(
            [NoSharding(), NoSharding(),
             NoSharding()], [Replicated(4)])
        self.run_resharding_task(src_shape,
                                 dst_shape,
                                 src_spec,
                                 dst_spec,
                                 tensor_shape,
                                 resharding_mode="broadcast")

        tensor_shape = (64, 64, 64)
        src_spec = ShardingSpec([Chunked(
            [2]), Chunked([2]), NoSharding()],
                                [ShardedAxis(0), ShardedAxis(1)])
        dst_spec = ShardingSpec([Chunked(
            [2]), NoSharding(), Chunked([2])],
                                [ShardedAxis(0), ShardedAxis(1)])
        self.run_resharding_task(src_shape,
                                 dst_shape,
                                 src_spec,
                                 dst_spec,
                                 tensor_shape,
                                 resharding_mode="broadcast")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ReshardingTest("test_4gpu_send_recv"))
    suite.addTest(ReshardingTest("test_4gpu_allgather"))
    suite.addTest(ReshardingTest("test_8gpu_2_dim_allgather"))
    suite.addTest(ReshardingTest("test_4gpu_broadcast"))
    suite.addTest(ReshardingTest("test_8gpu_broadcast"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
