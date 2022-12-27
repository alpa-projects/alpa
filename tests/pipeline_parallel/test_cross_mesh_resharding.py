"""Test cross-mesh resharding."""
import unittest
from alpa.pipeline_parallel.runtime_emitter import PipelineInstEmitter

import jax
from jax import xla
from jax.core import Var
from jax._src.abstract_arrays import ShapedArray
from jax.interpreters.pxla import (Chunked, NoSharding, Replicated, ShardedAxis,
                                   ShardingSpec, spec_to_indices)
import jax.numpy as jnp
import numpy as np

from alpa import init
from alpa.device_mesh import (DistributedArray, create_remote_array_refs,
                              get_global_virtual_physical_mesh)
from alpa.mesh_executable import next_mesh_executable_uuid
from alpa.global_env import global_config
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CollectiveGroup, ReshardingTaskSpec, CrossMeshCommunicator,
    SymbolicReshardingTask, SymbolicBroadcastReshardingTask)
from alpa.pipeline_parallel.pipeshard_executable import (
    AllocateZeroWorkerExecutableConfig, PipelineInstruction,
    PipeshardMeshWorkerExecutable)
from alpa.pipeline_parallel.resharding_tensor import VirtualDistributedArray
from alpa.testing import assert_allclose
from alpa.util import get_shard_shape


def test_resharding(var,
                    src_mesh,
                    src_sharding_spec,
                    dst_mesh,
                    dst_sharding_spec,
                    use_local_allgather,
                    resharding_mode,
                    src_loads=None,
                    dst_loads=None):
    global_config.use_local_allgather = use_local_allgather
    global_config.resharding_mode = resharding_mode

    # Resharding task spec and send/recv strategy
    src_loads = src_loads or {src: 0 for src in src_mesh.device_strs}
    dst_loads = dst_loads or {dst: 0 for dst in dst_mesh.device_strs}
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
        worker.put_executable.remote(exec_uuid, PipeshardMeshWorkerExecutable,
                                     instruction_lists[worker], [src_uuid], [],
                                     [], [], [],
                                     [False] * src_mesh.num_devices_per_host)
        exec_uuids[worker] = exec_uuid
    for worker in dst_mesh.workers:
        exec_uuid = next_mesh_executable_uuid()
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
    # for _ in range(3):
    # timers("overall_resharding_time").start()
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
    output_array = DistributedArray(dst_mesh, var.aval, dst_sharding_spec,
                                    output_refs[0])

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
                            use_local_allgather=True,
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
                        dst_sharding_spec, use_local_allgather, resharding_mode)
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
        # test allgather at the second dim
        tensor_shape = (3, 8, 2)
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                 tensor_shape)

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
    def test_8gpu_2_dim_allgather(self):
        src_shape = (1, 4)
        dst_shape = (1, 4)
        tensor_shape = (6, 8, 16)
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

    @unittest.skipIf(jax.device_count('gpu') < 8, "no enough device")
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
