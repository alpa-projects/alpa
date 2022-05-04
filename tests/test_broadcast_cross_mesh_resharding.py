import time
import unittest

from jax import xla
from jax.core import Var
from jax._src.abstract_arrays import ShapedArray
from jax.interpreters.pxla import (Chunked, NoSharding, Replicated, ShardedAxis,
                                   ShardingSpec, spec_to_indices)
import jax.numpy as jnp
import numpy as np
import ray

from alpa.device_mesh import DeviceCluster, DistributedArray, shard_arg_handlers
from alpa.mesh_executable import (create_remote_buffer_refs, get_uuid_np_array,
                                  next_mesh_executable_uuid)
from alpa.global_env import global_config
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CollectiveGroup, ReshardingTaskSpec, CrossMeshCommunicator,
    SymbolicReshardingTask, SymbolicBroadcastReshardingTask)
from alpa.pipeline_parallel.decentralized_distributed_runtime import (
    AllocateZeroWorkerExecutableConfig, DecentralizedDistributedRuntime,
    PipelineInstruction, PipelineMeshWorkerExecutable)
from alpa.pipeline_parallel.resharding_tensor import VirtualDistributedArray
from alpa.testing import assert_allclose
from alpa.util import get_ray_namespace_str, get_shard_shape
from alpa.timer import timers

def test_broadcast_resharding(var,
                    src_mesh,
                    src_sharding_spec,
                    dst_mesh,
                    dst_sharding_spec,
                    src_loads=None,
                    dst_loads=None):
    global_config.resharding_mode = "broadcast"
    src_loads = src_loads or {src: 0 for src in src_mesh.device_strs}
    dst_loads = dst_loads or {dst: 0 for dst in dst_mesh.device_strs}
    src_array = VirtualDistributedArray(device_mesh=src_mesh,
                                        aval=var.aval,
                                        sharding_spec=src_sharding_spec)
    dst_array = VirtualDistributedArray(device_mesh=dst_mesh,
                                        aval=var.aval,
                                        sharding_spec=dst_sharding_spec)
    task_spec = ReshardingTaskSpec(src_array, dst_array, None)
    strategy = CrossMeshCommunicator._generate_broadcast_resharding_strategy_by_loads(task_spec, src_loads, dst_loads)
    task_spec.set_resharding_strategy(strategy)
    # Resharding task. Compile send/recv from strategy and allgather.
    collective_group = CollectiveGroup(task_spec.get_participant_device_strs(),
                                       src_mesh, dst_mesh)
    if global_config.eagerly_create_communicators:
        collective_group.instantiate_now()
    else:
        collective_group.instantiate()
    task = SymbolicBroadcastReshardingTask(task_spec, collective_group, src_mesh, dst_mesh)
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
            PipelineInstruction.Run(config.exec_uuid,
                                    in_uuids,
                                    out_uuids, {
                                        "sync_before": False,
                                        "sync_after": False
                                    },
                                    info="allocate zero for recv"))
    # resharding task
    DecentralizedDistributedRuntime._compile_broadcast_resharding_task(
        src_mesh, dst_mesh, src_uuids, task, dst_uuids, instruction_lists)
    exec_uuids = {}
    # Compile Pipeline Executable
    for worker_idx, worker in enumerate(src_mesh.workers):
        exec_uuid = next_mesh_executable_uuid()
        worker.put_executable.remote(exec_uuid, PipelineMeshWorkerExecutable,
                                     instruction_lists[worker],
                                     [src_uuids[worker_idx]], [], [], [], [],
                                     [False] * src_mesh.num_devices_per_host)
        exec_uuids[worker] = exec_uuid
    for worker_idx, worker in enumerate(dst_mesh.workers):
        exec_uuid = next_mesh_executable_uuid()
        worker.put_executable.remote(exec_uuid, PipelineMeshWorkerExecutable,
                                     instruction_lists[worker], [],
                                     [dst_uuids[worker_idx]],
                                     executable_config_lists[worker], [], [],
                                     [False] * dst_mesh.num_devices_per_host)
        exec_uuids[worker] = exec_uuid

    for _ in range(5):
        # Prepare array and shard args
        test_array = np.arange(np.prod(var.aval.shape),
                            dtype=var.aval.dtype).reshape(var.aval.shape)
        indices = spec_to_indices(var.aval.shape, src_sharding_spec)
        test_array = xla.canonicalize_dtype(test_array)
        input_refs = shard_arg_handlers[type(test_array)](test_array, src_mesh,
                                                        indices)
        input_refs = np.array(input_refs).reshape(src_mesh.shape)
        input_uuids = get_uuid_np_array(input_refs)
        output_refs, output_uuids = create_remote_buffer_refs(dst_mesh)
        output_uuids = output_uuids.reshape(dst_mesh.shape)
        # Run executables
        timers("overall_resharding_time").start()
        for worker_idx, worker in enumerate(src_mesh.workers):
            worker.run_executable.remote(exec_uuids[worker],
                                        [input_uuids[worker_idx]], [])
        for worker_idx, worker in enumerate(dst_mesh.workers):
            worker.run_executable.remote(exec_uuids[worker], [],
                                        [output_uuids[worker_idx]])
        output_array = DistributedArray(dst_mesh, var.aval, dst_sharding_spec,
                                        output_refs)
        # Check correctness
        # assert_allclose(test_array, output_array._value)
        dst_mesh.sync_workers()
        timers("overall_resharding_time").stop()
        timers("overall_resharding_time").log()
        timers("overall_resharding_time").reset()

    # Delete executables
    for worker in src_mesh.workers:
        worker.delete_executable.remote(exec_uuids[worker])
    for worker in dst_mesh.workers:
        worker.delete_executable.remote(exec_uuids[worker])


class ReshardingTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto",
                 namespace=get_ray_namespace_str(
                     prefix=global_config.unittest_ray_namespace_prefix))

    def tearDown(self):
        ray.shutdown()
        time.sleep(1)

    def run_resharding_task(self,
                             src_mesh_shape,
                             dst_mesh_shape,
                             src_sharding_spec,
                             dst_sharding_spec,
                             tensor_shape,
                             tensor_dtype=None, 
                             src_mesh_logical_shape=None, 
                             dst_mesh_logical_shape=None):
        device_cluster = DeviceCluster()
        virtual_mesh = device_cluster.get_virtual_physical_mesh()
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

        # if src_mesh_logical_shape is not None:
        #     src_mesh = src_mesh.get_logical_mesh(src_mesh_logical_shape)
        # if dst_mesh_logical_shape is not None:
        #     dst_mesh = dst_mesh.get_logical_mesh(dst_mesh_logical_shape)
        tensor_dtype = tensor_dtype or jnp.int32
        var = Var(0, "", ShapedArray(tensor_shape, tensor_dtype))
        test_broadcast_resharding(var, src_mesh, src_sharding_spec, dst_mesh,
                                dst_sharding_spec)
        src_mesh.shutdown()
        dst_mesh.shutdown()

    def test_4gpu_broadcast_2(self):
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
        src_spec = ShardingSpec([Chunked(
            [2]), NoSharding(), NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                  tensor_shape)
        src_spec = ShardingSpec(
            [NoSharding(), Chunked([2]),
             NoSharding()], [ShardedAxis(0)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                  tensor_shape)

    def test_4gpu_broadcast_1(self):
        src_shape = (1, 4)
        dst_shape = (1, 4)
        base = 64
        tensor_shape = (8*base, 8*base, 16*base)#6*3+(3+3+4)=28
        src_spec = ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)])
        dst_spec = ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)])
        self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
                                  tensor_shape)

    # def test_4gpu_broadcast_3(self):
    #     src_shape = (1, 4)
    #     dst_shape = (2, 4)
    #     base = 64
    #     tensor_shape = (8*base, 8*base, 16*base)
    #     src_spec = ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)])
    #     dst_spec = ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(8)])
    #     self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
    #                               tensor_shape)

    # def test_4gpu_broadcast_4(self):
    #     src_shape = (1, 4)
    #     dst_shape = (2, 4)
    #     base = 64
    #     tensor_shape = (8*base, 8*base, 16*base)
    #     src_spec = ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)])
    #     dst_spec = ShardingSpec([NoSharding(), Chunked([2]), NoSharding()], [ShardedAxis(0)])
    #     self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
    #                               tensor_shape)

    # def test_4gpu_broadcast_5(self):
    #     src_shape = (1, 4)
    #     dst_shape = (2, 4)
    #     base = 64
    #     tensor_shape = (8*base, 8*base, 16*base)
    #     src_spec = ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)])
    #     dst_spec = ShardingSpec([NoSharding(), Chunked([2]), Chunked([2])], [ShardedAxis(0), ShardedAxis(1)])
    #     self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
    #                               tensor_shape)

    # def test_4gpu_broadcast_6(self):
    #     src_shape = (2, 4)
    #     dst_shape = (1, 4)
    #     base = 64
    #     tensor_shape = (8*base, 8*base, 16*base)
    #     src_spec = ShardingSpec([NoSharding(), Chunked([2]), Chunked([2])], [ShardedAxis(0), ShardedAxis(1)])
    #     dst_spec = ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)])
    #     self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
    #                               tensor_shape)

    # def test_4gpu_broadcast_7(self):
    #     src_shape = (1, 4)
    #     dst_shape = (1, 4)
    #     base = 64
    #     tensor_shape = (8*base, 8*base, 16*base)
    #     src_spec = ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)])
    #     dst_spec = ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)])
    #     self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
    #                               tensor_shape)

    # def test_4gpu_broadcast_8(self):
    #     src_shape = (1, 4)
    #     dst_shape = (1, 4)
    #     base = 64
    #     tensor_shape = (8*base, 8*base, 16*base)
    #     src_spec = ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)])
    #     dst_spec = ShardingSpec([Chunked([4]), NoSharding(), NoSharding()], [ShardedAxis(0)])
    #     self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
    #                               tensor_shape)

    # def test_4gpu_broadcast_9(self):
    #     src_shape = (1, 4)
    #     dst_shape = (1, 4)
    #     base = 64
    #     tensor_shape = (2, 8*base, 8*base*8*base)
    #     src_spec = ShardingSpec([Chunked([2]), Chunked([2]), NoSharding()], [ShardedAxis(0), ShardedAxis(1)])
    #     dst_spec = ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)])
    #     self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
    #                               tensor_shape, src_mesh_logical_shape=(2,2))

    # def test_4gpu_broadcast_9(self):
    #     src_shape = (1, 4)
    #     dst_shape = (1, 4)
    #     base = 64
    #     tensor_shape = (2, 8*base, 8*base*8*base)
    #     src_spec = ShardingSpec([Chunked([2]), Chunked([2]), NoSharding()], [ShardedAxis(0), ShardedAxis(1)])
    #     dst_spec = ShardingSpec([Chunked([2]), NoSharding(), Chunked([2])], [ShardedAxis(0), ShardedAxis(1)])
    #     self.run_resharding_task(src_shape, dst_shape, src_spec, dst_spec,
    #                               tensor_shape)

def suite():
    suite = unittest.TestSuite()
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_1"))
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_2"))
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_3"))
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_4"))
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_5"))
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_6"))
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_7"))
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_8"))
    # suite.addTest(ReshardingTest("test_4gpu_broadcast_9"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())