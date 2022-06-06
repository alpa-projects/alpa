"""The dirver part and worker part of a pipeshard executable."""
import logging
import time
from typing import Optional, Sequence, Callable

from jax.tree_util import tree_flatten, tree_unflatten
import numpy as np
import ray.exceptions

from alpa.device_mesh import MeshHostWorker
from alpa.global_env import global_config
from alpa.device_mesh import PhysicalDeviceMeshGroup
from alpa.mesh_executable import (AllocZeroBufferWorkerExecutable,
                                  ConcatMeshWorkerExecutable,
                                  MemzeroWorkerExecutable,
                                  PartialGradAccMeshWorkerExecutable,
                                  next_mesh_executable_uuid, get_uuid_np_array,
                                  next_remote_buffer_uuid, RemoteBufferRef)
from alpa.pipeline_parallel.runtime_emitter import (
    AllocateZeroWorkerExecutableConfig, ConcatWorkerExecutableConfig,
    ExecutableConfig, MemZeroWorkerExecutableConfig,
    PartialGradWorkerExecutableConfig, PipelineInstType, PipelineInstruction,
    PipeshardConfig)
from alpa.pipeline_parallel.schedules import PipelineSchedule
from alpa.pipeline_parallel.computation import XlaShardedPipelineComputation
from alpa.timer import timers
from alpa.util import OrderedSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

timer_names = {
    "overall": "average",
    "compute": "sum",
    "resharding_send": "sum",
    "resharding_recv": "sum",
    "free": "sum",
}


class PipeshardDriverExecutable:
    """The driver part of the executable for pipeshard parallel."""

    def __init__(self,
                 *,
                 stages: Sequence[XlaShardedPipelineComputation],
                 mesh_group: PhysicalDeviceMeshGroup,
                 pipeshard_config: PipeshardConfig,
                 schedule: PipelineSchedule,
                 is_batch: Sequence[bool],
                 num_batch: int,
                 flop_count: int,
                 static_argnums: Optional[Sequence[int]] = None,
                 out_tree_thunk: Optional[Callable] = None):
        ##### Input arguments #####
        self.stages = stages
        self.mesh_group = mesh_group
        self.schedule = schedule
        self.is_batch = is_batch
        self.num_batch = num_batch
        self.flop_count = flop_count
        self.num_mesh = len(mesh_group)
        self.static_argnums = static_argnums
        self.out_tree_thunk = out_tree_thunk

        ##### For debugging #####
        # List[stage_idx -> str]
        self.hlo_texts_after_spmd_partitioner = []

        # Compile pipeline instructions and configs of mesh executables
        self._instantiate_nccl_groups(pipeshard_config.device_str_groups)

        ##### Internal states #####

        # List[stage_idx -> executable_uuid]
        self.executable_uuids = pipeshard_config.executable_uuids
        ##### For handling inputs of the executable ####
        # Whether the var should be donated
        # List[mesh_idx -> List[bool]]
        self.donate_invars = pipeshard_config.donate_invars
        # List[mesh_idx -> List[arg_idx]]
        self.mesh_arg_indices = pipeshard_config.mesh_arg_indices
        # Cached sharding indices for input arguments
        # List[mesh_idx -> List[sharding_indices]].
        self.input_shard_indices = pipeshard_config.input_shard_indices
        # Whether the argument should be deleted after shard
        # List[mesh_idx -> List[bool]]
        self.delete_after_shard = pipeshard_config.delete_after_shard
        # Whether the argument is a batch argument
        # List[mesh_idx -> List[bool]]
        self.batch_invars = pipeshard_config.batch_invars

        ##### For handling outputs of the executable ####
        # Dict[worker -> List[uuid]]
        self.output_local_uuid_list = pipeshard_config.output_local_uuid_list

        # For handling input/outputs
        self.outs_handler: Callable = pipeshard_config.outs_handler

        # For weight serialization
        self.load_info = pipeshard_config.load_info

        self.resharding_tasks = pipeshard_config.resharding_tasks

        if global_config.eagerly_create_communicators:
            for task in self.resharding_tasks:
                task.create_resharding_communicators()

        # Create a PipeshardMeshWorkerExecuable for each MeshHostWorker
        self.worker_executable_uuid_mapping = {}  # Dict[
        for mesh_idx, physical_mesh in enumerate(self.mesh_group):
            mesh_grad_uuids = pipeshard_config.grad_uuids[mesh_idx]
            for worker_idx, worker in enumerate(physical_mesh.workers):
                acc_grad_local_uuids = []
                if len(mesh_grad_uuids) > 0:
                    acc_grad_local_uuids = mesh_grad_uuids[worker_idx]
                args = (pipeshard_config.instruction_lists[worker],
                        pipeshard_config.input_local_uuid_lists[worker],
                        self.output_local_uuid_list[worker],
                        pipeshard_config.executable_configs[worker],
                        acc_grad_local_uuids,
                        pipeshard_config.reduced_var_uuid_lists[worker],
                        self.donate_invars[mesh_idx])
                uuid = next_mesh_executable_uuid()
                worker.put_executable.remote(uuid, PipeshardMeshWorkerExecuable,
                                             *args)
                self.worker_executable_uuid_mapping[worker] = uuid

    ##### Compilation Related Functions #####

    def _instantiate_nccl_groups(self, device_str_groups):
        """
        Instantiate NCCL groups between two physical meshes.

        Args:
            device_str_groups (List[List[set]]): a num_mesh x num_mesh matrix.
                Only entries at device_str_groups[i][j] (i < j) are filled,
                entries with i > j are None, because (spec[i][j], spec[j][i])
                will share collective groups.
        """

        # construct groups
        start_time = time.time()
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if i >= j:
                    assert not device_str_groups[i][j]
                    continue
                if not device_str_groups[i][j]:
                    continue
                self.mesh_group.instantiate_nccl_group(i, j)
        end_time = time.time()
        logger.debug(
            f"Initialize collective group takes {end_time - start_time:.2f}")

    ##### Execution Related Functions #####
    def launch_on_driver(self, *args):
        """Launch the executable on the driver.

        Args:
            args: The original arguments of the parallelized function.
        """
        input_bufs = [None for _ in range(self.num_mesh)]
        output_bufs = [None for _ in range(self.num_mesh)]
        output_uuids = [None for _ in range(self.num_mesh)]

        num_outs = [
            len(self.output_local_uuid_list[mesh.workers[0]])
            for mesh in self.mesh_group
        ]

        for mesh_idx, physical_mesh in enumerate(self.mesh_group):
            # Shard inputs
            mesh_args = [args[idx] for idx in self.mesh_arg_indices[mesh_idx]]
            tmp_bufs = physical_mesh.shard_args_to_bufs(
                self.input_shard_indices[mesh_idx],
                self.delete_after_shard[mesh_idx], self.batch_invars[mesh_idx],
                self.num_batch, mesh_args)

            # Flatten the batch args in tmp_bufs
            flatten_bufs = []
            for i, is_batch_invar in enumerate(self.batch_invars[mesh_idx]):
                if is_batch_invar:
                    flatten_bufs.extend(tmp_bufs[i])
                else:
                    flatten_bufs.append(tmp_bufs[i])
            input_bufs[mesh_idx] = flatten_bufs

            # Convert bufs to uuids
            num_hosts = physical_mesh.num_hosts
            num_devices_per_host = physical_mesh.num_devices_per_host
            input_uuids = get_uuid_np_array(input_bufs[mesh_idx]).reshape(
                (-1, num_hosts, num_devices_per_host)).transpose([1, 0, 2])
            output_uuids[mesh_idx] = next_remote_buffer_uuid(
                num_hosts * num_outs[mesh_idx] * num_devices_per_host).reshape(
                    num_hosts, num_outs[mesh_idx], num_devices_per_host)

            # Execute
            for i, worker in enumerate(physical_mesh.workers):
                worker.run_executable.remote(
                    self.worker_executable_uuid_mapping[worker],
                    input_uuids[i],
                    output_uuids[mesh_idx][i],
                    sync_for_timer=global_config.pipeline_sync_for_timer)

        # Handle donation
        for mesh_idx in range(len(self.mesh_group)):
            inputs = input_bufs[mesh_idx]
            for bufs, donate in zip(inputs, self.donate_invars[mesh_idx]):
                if donate:
                    for buf in bufs:
                        buf.set_deleted_on_workers()

        # Construct output_bufs
        for mesh_idx, physical_mesh in enumerate(self.mesh_group):
            num_devices_per_host = physical_mesh.num_devices_per_host
            output_uuid_transposed = output_uuids[mesh_idx].transpose([1, 0, 2])
            output_bufs[mesh_idx] = np.empty(
                (num_outs[mesh_idx], physical_mesh.num_devices), dtype=object)
            for i in range(num_outs[mesh_idx]):
                for j in range(physical_mesh.num_devices):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[mesh_idx][i][j] = RemoteBufferRef(
                        physical_mesh, host_id, device_id,
                        output_uuid_transposed[i][host_id][device_id])

        # Check if there is OOM
        if global_config.pipeline_check_alive:
            self._check_alive()

        return self.outs_handler(self.mesh_group, output_bufs)

    def __call__(self, *args):
        """Fast call without signature matching."""
        if self.static_argnums:
            dyn_args = [
                args[i]
                for i in range(len(args))
                if i not in self.static_argnums
            ]
        else:
            dyn_args = args
        args_flat, _ = tree_flatten(dyn_args)
        out = self.launch_on_driver(*args_flat)
        return tree_unflatten(self.out_tree_thunk(), out)

    ##### Load/Store Related Functions #####
    def get_load_info(self):
        return self.load_info

    ##### Profiling and Debugging Related Functions #####
    def get_execution_time_costs(self,
                                 warmup=2,
                                 timer_name="overall",
                                 return_all_costs=False):
        """Get the execution time costs with internal timers."""
        if timer_name not in timer_names:
            raise RuntimeError(
                f"Unrecognized timer name for pipeline parallel runtime. "
                f"Query timer name from the following: {timer_names.keys()}.")
        mesh_costs = []
        for mesh in self.mesh_group:
            mesh_costs.append(mesh.get_remote_timer(timer_name).costs[warmup:])
        if return_all_costs:
            return mesh_costs

        min_costs = [1.0e9] * len(mesh_costs[0])
        max_costs = [0] * len(mesh_costs[0])
        for mesh_cost in mesh_costs:
            for i, cost in enumerate(mesh_cost):
                if cost > max_costs[i]:
                    max_costs[i] = cost
                if cost < min_costs[i]:
                    min_costs[i] = cost
        return max_costs

    def reset_benchmark_timers(self):
        """Reset all benchmarking timers."""
        for name in timer_names:
            for mesh in self.mesh_group:
                mesh.reset_remote_timer(name)

    def get_hlo_text(self, after_spmd_partitioner=True):
        """Return the HLO text for all stages."""
        if after_spmd_partitioner:
            if self.hlo_texts_after_spmd_partitioner:
                return self.hlo_texts_after_spmd_partitioner

            hlo_texts = []
            for stage_idx in range(len(self.stages)):
                mesh_idx = self.schedule.stage_placement(stage_idx)
                assert len(mesh_idx) == 1
                mesh_idx = list(mesh_idx)[0]
                physical_mesh = self.mesh_group[mesh_idx]
                hlo_text = physical_mesh.workers[0].get_exec_hlo_text.remote(
                    self.executable_uuids[stage_idx])
                hlo_texts.append(hlo_text)
            self.hlo_texts_after_spmd_partitioner = ray.get(hlo_texts)
            return self.hlo_texts_after_spmd_partitioner
        else:
            ret = []
            for stage in self.stages:
                ret.append(stage.get_hlo_text())
            return ret

    def get_total_allocation_size(self):
        """Get the total allocated memory size of each mesh."""
        # TODO: compute the theoretical total allocation size
        raise NotImplementedError()

    def profile_all_executable_with_dummy_inputs(self):
        """Profile all stage executables with dummy inputs."""
        all_profiled_handles = []
        for _, physical_mesh in enumerate(self.mesh_group):
            all_worker_profiled = []
            for _, worker in enumerate(physical_mesh.workers):
                worker: MeshHostWorker
                all_worker_profiled.append(
                    worker.profile_executable_with_dummy_inputs.remote(
                        self.worker_executable_uuid_mapping[worker]))
            if len(all_worker_profiled) == 1:
                all_worker_profiled = all_worker_profiled[0]
            all_profiled_handles.append(all_worker_profiled)
        all_profiled = [ray.get(handles) for handles in all_profiled_handles]
        return all_profiled

    def print_resharding_tasks(self):
        """Pretty print all compiled resharding tasks."""
        ret = ""
        for task in self.resharding_tasks:
            ret += str(task) + "\n\n"
        return ret

    def _debug_check(self):
        for mesh in self.mesh_group:
            num_outs = -1
            for worker in mesh.workers:
                if num_outs == -1:
                    num_outs = len(self.output_local_uuid_list[worker])
                else:
                    assert len(self.output_local_uuid_list[worker]) == num_outs

    ##### Other Functions #####
    def sync(self):
        """Sync device activities on all workers."""
        self.mesh_group.sync_workers()

    def _check_alive(self):
        try:
            rets = [
                worker.check_alive.remote()
                for mesh in self.mesh_group
                for worker in mesh.workers
            ]
            ray.get(rets)
        except ray.exceptions.RayActorError:
            self.mesh_group.exception_shutdown()

    def __del__(self):
        for worker, uuid in self.worker_executable_uuid_mapping.items():
            worker.delete_executable.remote(uuid)


class PipeshardMeshWorkerExecuable:
    """An executable that executes static pipeline runtime instructions on a
    worker."""

    def __init__(self, worker: MeshHostWorker, uuid: int,
                 instructions: Sequence[PipelineInstruction],
                 input_local_uuids: Sequence[int],
                 output_local_uuids: Sequence[int],
                 executable_configs: Sequence[ExecutableConfig],
                 acc_local_uuids: np.ndarray, acc_out_uuids: np.ndarray,
                 donate_invars: Sequence[bool]):
        # Instruction Lists
        self.my_uuid = uuid
        self.instructions = instructions
        self.input_local_uuids = input_local_uuids
        self.output_local_uuids = output_local_uuids
        self.donate_invars = donate_invars

        # Buffer management
        self.worker = worker
        self.global_buffers = worker.buffers
        self.acc_grad_buffers = {}
        self.acc_in_uuids = acc_local_uuids
        self.acc_out_uuids = acc_out_uuids

        # Executable management
        self._related_exec_uuids = []
        self.partial_grad_exec_uuids = OrderedSet()
        self.use_memzero = False

        # Create tasks
        for task_config in executable_configs:
            self._related_exec_uuids.append(task_config.exec_uuid)
            if isinstance(task_config, PartialGradWorkerExecutableConfig):
                self.worker.put_executable(task_config.exec_uuid,
                                           PartialGradAccMeshWorkerExecutable,
                                           task_config.hlo_proto,
                                           task_config.strategy_config,
                                           task_config.grad_sync_channel_ids)
                self.partial_grad_exec_uuids.add(task_config.exec_uuid)
            elif isinstance(task_config, MemZeroWorkerExecutableConfig):
                assert len(self.acc_grad_buffers) == 0
                # allocate buffers
                self.use_memzero = True
                self.worker.put_executable(task_config.exec_uuid,
                                           AllocZeroBufferWorkerExecutable,
                                           task_config.grad_shard_shapes,
                                           task_config.grad_shard_dtypes)
                self.worker.buffers = self.acc_grad_buffers
                self.worker.run_executable(task_config.exec_uuid, [],
                                           acc_local_uuids)
                self.worker.buffers = self.global_buffers
                self.worker.delete_executable(task_config.exec_uuid)
                # replace the temp AllocZeroExecutable by Memzero ones
                self.worker.put_executable(task_config.exec_uuid,
                                           MemzeroWorkerExecutable,
                                           task_config.grad_shard_shapes,
                                           task_config.grad_shard_dtypes)
            elif isinstance(task_config, AllocateZeroWorkerExecutableConfig):
                self.worker.put_executable(task_config.exec_uuid,
                                           AllocZeroBufferWorkerExecutable,
                                           task_config.grad_shard_shapes,
                                           task_config.grad_shard_dtypes)
            elif isinstance(task_config, ConcatWorkerExecutableConfig):
                self.worker.put_executable(task_config.exec_uuid,
                                           ConcatMeshWorkerExecutable,
                                           *task_config[1:])
            else:
                raise ValueError(f"Invalid task config {task_config}")
        self.partial_grad_exec_uuids = list(self.partial_grad_exec_uuids)

    def execute_on_worker(self, input_global_uuids, output_global_uuids,
                          sync_for_timer):
        """Execute on the mesh worker given input and output uuids."""
        # create a local buffer environment
        assert len(self.input_local_uuids) == len(input_global_uuids)
        buffers = {}
        for local_ids, global_ids in zip(self.input_local_uuids,
                                         input_global_uuids):
            for local_id, global_id in zip(local_ids, global_ids):
                buffers[local_id] = self.global_buffers[global_id]
        # add preallocated buffers for gradient accumulation
        buffers.update(self.acc_grad_buffers)
        # donate invars
        for global_ids, donate in zip(input_global_uuids, self.donate_invars):
            if donate:
                self.worker.delete_buffers(global_ids)
        # load the local env
        self.worker.buffers = buffers
        sync_func = self.worker.sync if sync_for_timer else None

        # Execute
        timers("overall").start(sync_func=sync_func)
        for instruction in self.instructions:
            # print(f"memory_allocated: "
            #       f"{self.worker.get_memory_allocated()/1024**3:.3f} GB  "
            #       f"max_memory_allocated: "
            #       f"{self.worker.get_max_memory_allocated()/1024**3:.3f} GB "
            #       f"next instruction: {instruction}")
            if instruction.opcode == PipelineInstType.RUN:
                timers("compute").start()
                self.worker.run_executable(instruction.task_uuid,
                                           instruction.input_uuids,
                                           instruction.output_uuids,
                                           **instruction.opaques["kwargs"])
                timers("compute").suspend()
            elif instruction.opcode == PipelineInstType.SEND:
                timers("resharding_send").start()
                self.worker.run_resharding_send_task(instruction.task_uuid,
                                                     instruction.input_uuids)
                timers("resharding_send").suspend()
            elif instruction.opcode == PipelineInstType.RECV:
                timers("resharding_recv").start()
                self.worker.run_resharding_recv_task(
                    instruction.task_uuid, instruction.output_uuids,
                    instruction.opaques["set_empty_buffer"])
                # TODO(lmzheng): move this to run_resharding_recv_task
                if instruction.opaques["allgather_uuid"] is not None:
                    self.worker.run_allgather_task(
                        instruction.opaques["allgather_uuid"],
                        instruction.output_uuids)
                timers("resharding_recv").suspend()
            elif instruction.opcode == PipelineInstType.BROADCAST:
                timers("resharding_broadcast").start()
                self.worker.run_resharding_broadcast_task(
                    instruction.task_uuid,
                    instruction.input_uuids if instruction.input_uuids
                    is not None else instruction.output_uuids)
                timers("resharding_broadcast").suspend()
            elif instruction.opcode == PipelineInstType.FREE:
                timers("free").start()
                self.worker.delete_buffers(instruction.input_uuids)
                timers("free").suspend()

        for timer_name in [
                "compute", "resharding_send", "resharding_recv",
                "resharding_broadcast", "free"
        ]:
            if timer_name in timers:
                timers(timer_name).stop()
                # timers(timer_name).log(mode="sum")
                # timers(timer_name).reset()
        timers("overall").stop(sync_func=sync_func)

        # copy to global env
        assert len(self.output_local_uuids) == len(output_global_uuids)
        for local_ids, global_ids in zip(self.output_local_uuids,
                                         output_global_uuids):
            for local_id, global_id in zip(local_ids, global_ids):
                self.global_buffers[global_id] = buffers[local_id]
        # now acc_grad_buffers are those after grad acc, before apply grad
        # with memzero. These buffers are reused in the next iteration.
        # TODO(yonghao): never donate them
        if self.use_memzero:
            for in_uuids, out_uuids in zip(self.acc_in_uuids,
                                           self.acc_out_uuids):
                for in_uuid, out_uuid in zip(in_uuids, out_uuids):
                    self.acc_grad_buffers[in_uuid] = buffers[out_uuid]
        # restore global environment
        self.worker.buffers = self.global_buffers
        buffers.clear()

    def profile_with_dummy_inputs(self):
        """Profile the executable with dummy inputs."""
        self.worker.reset_memory_stats()
        ret = {
            exec_id:
            (np.mean(
                self.worker.profile_executable_with_dummy_inputs(
                    exec_id, skip_grad_sync=False)),
             self.worker.get_exec_total_allocation_size(exec_id) / 1024**3)
            for exec_id in self.partial_grad_exec_uuids
        }
        self.worker.reset_memory_stats()
        return ret

    def __del__(self):
        for exec_id in self._related_exec_uuids:
            self.worker.delete_executable(exec_id)
