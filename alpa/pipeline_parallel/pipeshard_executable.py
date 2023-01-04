"""The dirver part and worker part of a pipeshard executable."""
import logging
from functools import partial
import json
import os
import time
from typing import Optional, Sequence

from jax._src import traceback_util
from jax._src.lib import xla_extension as xe
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves, PyTreeDef
import numpy as np
import ray.exceptions

from alpa.device_mesh import (
    MeshHostWorker, RemoteArrayRef,
    create_and_record_cross_mesh_collective_communicators, next_array_uuids)
from alpa.global_env import global_config
from alpa.device_mesh import PhysicalDeviceMeshGroup
from alpa.mesh_executable import (AllocZeroBufferWorkerExecutable,
                                  UtilMeshWorkerExecutable,
                                  PartialGradAccMeshWorkerExecutable,
                                  next_mesh_executable_uuid,
                                  get_execution_timer_name)
from alpa.parallel_plan import ClusterInfo, PipelinePlan, ParallelPlan
from alpa.pipeline_parallel.layer_construction import LayerOption
from alpa.pipeline_parallel.runtime_emitter import (
    AllocateZeroWorkerExecutableConfig, ConcatWorkerExecutableConfig,
    ExecutableConfig, PartialGradWorkerExecutableConfig, PipelineInstType,
    PipelineInstruction, PipeshardConfig)
from alpa.shard_parallel.auto_sharding import HloStatus
from alpa.timer import timers, tracer
from alpa.util import OrderedSet

traceback_util.register_exclusion(__file__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipeshardDriverExecutable:
    """The driver part of the executable for pipeshard parallel."""

    def __init__(self,
                 mesh_group: PhysicalDeviceMeshGroup,
                 pipeshard_config: PipeshardConfig,
                 num_batch: int,
                 layer_option: LayerOption,
                 in_tree: PyTreeDef,
                 out_tree: Optional[PyTreeDef] = None,
                 static_argnums: Optional[Sequence[int]] = None):
        ##### Input arguments #####
        self.mesh_group = mesh_group
        self.num_mesh = len(mesh_group)
        self.num_batch = num_batch
        self.in_tree = in_tree
        self.out_tree = out_tree
        self.static_argnums = static_argnums

        ##### For debugging and serialization #####
        self.stages = pipeshard_config.xla_stages
        self.schedule = pipeshard_config.schedule
        self.flop_count = pipeshard_config.flop_count
        self.stage_input_shard_specs = pipeshard_config.stage_input_shard_specs
        self.input_placement_specs = pipeshard_config.input_placement_specs
        self.output_placement_specs = pipeshard_config.output_placement_specs
        # List[stage_idx -> str]
        self.fully_optimized_hlo_texts = []
        # List[stage_idx -> int]
        self.stage_allocation_sizes = []
        self.sharding_annotated_hlo_texts = (
            pipeshard_config.sharding_annotated_hlo_texts)
        # List[stage_idx -> executable_uuid]
        self.executable_uuids = pipeshard_config.executable_uuids
        self.default_auto_sharding_option = (
            pipeshard_config.default_auto_sharding_option)
        self.pipeline_plan = PipelinePlan(
            self.schedule.name,
            layer_option,
            pipeshard_config.manual_stage_option,
        )

        ##### For handling inputs of the executable #####
        # go to the definition of PipeshardInputConfig for more details.
        input_config = pipeshard_config.input_config
        self.donate_invars = input_config.donate_invars
        self.mesh_arg_indices = input_config.mesh_arg_indices
        self.input_shard_indices = input_config.input_shard_indices
        self.delete_after_shard = input_config.delete_after_shard
        self.batch_invars = input_config.batch_invars

        ##### For handling outputs of the executable #####
        self.output_local_uuid_list = pipeshard_config.output_local_uuid_list
        self.outs_handler = pipeshard_config.outs_handler

        ##### For cross-mesh resharding #####
        self._instantiate_nccl_groups(pipeshard_config.device_str_groups)
        self.resharding_tasks = pipeshard_config.resharding_tasks
        for mesh_ids in pipeshard_config.allreduce_groups:
            meshes = [self.mesh_group.meshes[idx] for idx in mesh_ids]
            create_and_record_cross_mesh_collective_communicators(meshes)
        if global_config.eagerly_create_communicators:
            for task in self.resharding_tasks:
                task.create_resharding_communicators()

        self.exec_uuid = next_mesh_executable_uuid()
        # Create a PipeshardMeshWorkerExecutable for each MeshHostWorker
        for mesh_idx, physical_mesh in enumerate(self.mesh_group):
            mesh_grad_uuids = pipeshard_config.grad_uuids[mesh_idx]
            for worker in physical_mesh.workers:
                acc_grad_local_uuids = []
                if len(mesh_grad_uuids) > 0:
                    acc_grad_local_uuids = mesh_grad_uuids
                args = (pipeshard_config.instruction_lists[worker],
                        input_config.input_local_uuid_lists[mesh_idx],
                        self.output_local_uuid_list[mesh_idx],
                        pipeshard_config.executable_configs[worker],
                        acc_grad_local_uuids,
                        pipeshard_config.reduced_var_uuid_lists[mesh_idx],
                        self.donate_invars[mesh_idx])
                worker.put_executable.remote(self.exec_uuid,
                                             PipeshardMeshWorkerExecutable,
                                             *args)

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
        start_time = time.time()
        for i in range(self.num_mesh):
            for j in range(i, self.num_mesh):
                if device_str_groups[i][j]:
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
            len(self.output_local_uuid_list[mesh_idx])
            for mesh_idx in range(self.num_mesh)
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
            input_uuids = np.array([ref.uuid for ref in input_bufs[mesh_idx]])
            output_uuids[mesh_idx] = next_array_uuids(num_outs[mesh_idx])

            # Execute
            for worker in physical_mesh.workers:
                worker.run_executable.remote(
                    self.exec_uuid,
                    input_uuids,
                    output_uuids[mesh_idx],
                    sync_for_timer=global_config.pipeline_sync_for_timer,
                    collect_trace=global_config.collect_trace)

        # Handle donation
        for mesh_idx in range(len(self.mesh_group)):
            inputs = input_bufs[mesh_idx]
            for ref, donate in zip(inputs, self.donate_invars[mesh_idx]):
                if donate:
                    ref.set_deleted_on_workers()

        # Construct output_bufs
        for mesh_idx, physical_mesh in enumerate(self.mesh_group):
            output_uuid = output_uuids[mesh_idx]
            output_bufs[mesh_idx] = np.empty((num_outs[mesh_idx],),
                                             dtype=object)
            for i in range(num_outs[mesh_idx]):
                output_bufs[mesh_idx][i] = RemoteArrayRef(
                    physical_mesh, output_uuid[i])

        # Check if there is OOM
        if global_config.pipeline_check_alive:
            self._check_alive()

        return self.outs_handler(self.mesh_group, output_bufs)

    def get_input_placement_specs(self):
        """
        Return the preferred placement specs for input arguments.
        The return value is a pytree of PlacementSpec
        with the same structure as the input pytree.
        """
        return tree_unflatten(self.in_tree, self.input_placement_specs)

    def get_output_placement_specs(self):
        """
        Return the preferred placement specs for outputs.
        The return value is a pytree of PlacementSpec
        with the same structure as the output pytree.
        """
        return tree_unflatten(self.out_tree, self.output_placement_specs)

    def get_parallel_plan(self):
        """Get the overall parallel plan."""
        virtual_mesh = self.mesh_group.parent
        cluster_info = ClusterInfo(virtual_mesh.num_hosts,
                                   virtual_mesh.num_devices_per_host)
        return ParallelPlan(cluster_info, self.num_batch,
                            self.default_auto_sharding_option,
                            self.pipeline_plan,
                            tree_leaves(self.get_input_placement_specs()))

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
        return tree_unflatten(self.out_tree, out)

    ##### Profiling and Debugging Related Functions #####
    def get_stage_execution_info(self):
        """Get the per-stage execution information of all invocations.
           Return a list, where each element corresponds to a single stage.
           Each element is a list of (start, stop, node_ids, devices) tuple,
           where each tuple corresponds to one invocation.
        """
        exec_timer_name = get_execution_timer_name(self.exec_uuid)
        run_begin_event = exec_timer_name + "-ins-run-begin"
        run_end_event = exec_timer_name + "-ins-run-end"

        num_stages = len(self.stages)
        stage_start = [[] for _ in range(num_stages)]
        stage_end = [[] for _ in range(num_stages)]

        # Extract events
        for mesh in self.mesh_group:
            mesh_tracer = mesh.get_remote_tracer()

            for x in mesh_tracer.events:
                if x.name == run_begin_event and "stage" in x.info:
                    stage_id = int(x.info[6:])
                    stage_start[stage_id].append(x.tstamp)
                if x.name == run_end_event and "stage" in x.info:
                    stage_id = int(x.info[6:])
                    stage_end[stage_id].append(x.tstamp)

        # Organize return values
        all_stages_info_list = []
        for i in range(num_stages):
            mesh_idx = self.schedule.stage_placement(i)
            assert len(mesh_idx) == 1
            mesh_idx = list(mesh_idx)[0]
            mesh = self.mesh_group[mesh_idx]
            host_ids, devices = mesh.host_ids, mesh.devices
            per_stage_info_list = []
            for s, e in zip(stage_start[i], stage_end[i]):
                per_stage_info_list.append((s, e, host_ids, devices))
            all_stages_info_list.append(per_stage_info_list)
        return all_stages_info_list

    def get_execution_time_costs(self, timer_name=None, return_all_costs=False):
        """Get the execution time costs with internal timers."""
        assert timer_name is None  # TODO(lmzheng): support other timers later
        timer_name = get_execution_timer_name(self.exec_uuid)
        mesh_costs = []
        for mesh in self.mesh_group:
            mesh_costs.append(mesh.get_remote_timer(timer_name).costs)
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

    def get_shard_args_time_costs(self):
        # TODO(lmzheng): implement this
        raise NotImplementedError()

    def get_hlo_text(self, status: HloStatus = HloStatus.FULLY_OPTIMIZED):
        """Return the HLO text for all stages."""
        if status == HloStatus.FULLY_OPTIMIZED:
            if self.fully_optimized_hlo_texts:
                return self.fully_optimized_hlo_texts

            hlo_texts = []
            for stage_idx in range(len(self.stages)):
                mesh_idx = self.schedule.stage_placement(stage_idx)
                assert len(mesh_idx) == 1
                mesh_idx = list(mesh_idx)[0]
                physical_mesh = self.mesh_group[mesh_idx]
                hlo_text = physical_mesh.workers[0].get_exec_hlo_text.remote(
                    self.executable_uuids[stage_idx])
                hlo_texts.append(hlo_text)
            self.fully_optimized_hlo_texts = ray.get(hlo_texts)
            return self.fully_optimized_hlo_texts
        else:
            return self.sharding_annotated_hlo_texts

    def get_stage_allocation_size(self):
        """Get the total memory allocation size in bytes of all stages."""
        if self.stage_allocation_sizes:
            return self.stage_allocation_sizes

        sizes = []
        for stage_idx in range(len(self.stages)):
            mesh_idx = self.schedule.stage_placement(stage_idx)
            assert len(mesh_idx) == 1
            mesh_idx = list(mesh_idx)[0]
            physical_mesh = self.mesh_group[mesh_idx]
            size = physical_mesh.workers[
                0].get_exec_total_allocation_size.remote(
                    self.executable_uuids[stage_idx])
            sizes.append(size)
        self.stage_allocation_sizes = ray.get(sizes)
        return self.stage_allocation_sizes

    def dump_debug_info(self, folder: str):
        """
        Dump intermediate representations and other informations for debugging.
        """
        os.makedirs(folder, exist_ok=True)
        name = self.stages[0].hlo.name
        name = name[:name.index("pipeshard_parallel") - 1]
        prefix = os.path.join(folder, name)

        fully_optimized_hlo_texts = self.get_hlo_text(HloStatus.FULLY_OPTIMIZED)
        allocation_sizes = self.get_stage_allocation_size()
        for stage_idx in range(len(self.stages)):
            with open(f"{prefix}_stage_{stage_idx}.hlo", "w") as f:
                f.write(fully_optimized_hlo_texts[stage_idx])

            with open(f"{prefix}_stage_{stage_idx}.mem_usage.txt", "w") as f:
                f.write(f"total_allocation_size: "
                        f"{allocation_sizes[stage_idx]/(1024**3):.3f} GB\n")

        with open(f"{prefix}_resharding_tasks.txt", "w") as f:
            for task in self.resharding_tasks:
                f.write(str(task) + "\n\n")

        with open(f"{prefix}_input_placement_specs.txt", "w") as f:
            f.write(str(self.get_input_placement_specs()))
        with open(f"{prefix}_output_placement_specs.txt", "w") as f:
            f.write(str(self.get_output_placement_specs()))

    def dump_stage_execution_trace(self, filename: str):
        exec_info = self.get_stage_execution_info()
        dump_stage_execution_trace_internal(exec_info, filename)

    def profile_all_executable_with_dummy_inputs(self):
        """Profile all stage executables with dummy inputs."""
        all_profiled_handles = []
        for _, physical_mesh in enumerate(self.mesh_group):
            all_worker_profiled = []
            for _, worker in enumerate(physical_mesh.workers):
                worker: MeshHostWorker
                all_worker_profiled.append(
                    worker.profile_executable_with_dummy_inputs.remote(
                        self.exec_uuid))
            if len(all_worker_profiled) == 1:
                all_worker_profiled = all_worker_profiled[0]
            all_profiled_handles.append(all_worker_profiled)
        all_profiled = [ray.get(handles) for handles in all_profiled_handles]
        return all_profiled

    ##### Other Functions #####
    def sync(self):
        """Sync device activities on all workers."""
        self.mesh_group.sync_workers()

    def sync_move_workers(self):
        """Sync moveworkers on all meshes."""
        self.mesh_group.sync_move_workers()

    def _check_alive(self):
        """
        Check whether all workers are alive.
        Shutdown the runtime if any worker dies.
        """
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
        for mesh in self.mesh_group:
            mesh.delete_remote_executable(self.exec_uuid)


class PipeshardMeshWorkerExecutable:
    """
    An executable that executes static pipeline runtime instructions on a
    worker.
    """

    def __init__(self, worker: MeshHostWorker, uuid: int,
                 instructions: Sequence[PipelineInstruction],
                 input_local_uuids: Sequence[int],
                 output_local_uuids: Sequence[int],
                 executable_configs: Sequence[ExecutableConfig],
                 acc_local_uuids: np.ndarray, acc_out_uuids: np.ndarray,
                 donate_invars: Sequence[bool]):
        # Instruction Lists
        self.exec_uuid = uuid
        self.exec_timer_name = get_execution_timer_name(uuid)
        self.instructions = instructions
        self.input_local_uuids = input_local_uuids
        self.output_local_uuids = output_local_uuids

        # Buffer management
        self.worker = worker
        self.global_buffers = worker.buffers
        self.acc_in_uuids = acc_local_uuids
        self.acc_out_uuids = acc_out_uuids
        self.donate_invars = donate_invars

        # Executable management
        self._related_exec_uuids = []
        self.partial_grad_exec_uuids = OrderedSet()

        # Compile executables
        for task_config in executable_configs:
            self._related_exec_uuids.append(task_config.exec_uuid)
            if isinstance(task_config, PartialGradWorkerExecutableConfig):
                self.worker.put_executable(task_config.exec_uuid,
                                           PartialGradAccMeshWorkerExecutable,
                                           *task_config[1:])
                self.partial_grad_exec_uuids.add(task_config.exec_uuid)
            elif isinstance(task_config, AllocateZeroWorkerExecutableConfig):
                self.worker.put_executable(task_config.exec_uuid,
                                           AllocZeroBufferWorkerExecutable,
                                           task_config.grad_shard_shapes,
                                           task_config.grad_shard_dtypes)
            elif isinstance(task_config, ConcatWorkerExecutableConfig):
                self.worker.put_executable(task_config.exec_uuid,
                                           UtilMeshWorkerExecutable,
                                           *task_config[1:])
            else:
                raise ValueError(f"Invalid task config {task_config}")
        self.partial_grad_exec_uuids = list(self.partial_grad_exec_uuids)

    def execute_on_worker(self, input_global_uuids, output_global_uuids,
                          sync_for_timer, collect_trace):
        """Execute on the mesh worker given input and output uuids."""
        # create a local buffer environment
        assert len(self.input_local_uuids) == len(input_global_uuids)
        buffers = {}
        for local_id, global_id in zip(self.input_local_uuids,
                                       input_global_uuids):
            buffers[local_id] = self.global_buffers[global_id]
        if global_config.enable_overlapping:
            xe.reset_event_context(self.worker.backend)
        # donate invars
        for global_id, donate in zip(input_global_uuids, self.donate_invars):
            if donate:
                self.worker.delete_buffers(global_id)
        # load the local env
        self.worker.buffers = buffers
        sync_func = self.worker.sync if sync_for_timer else None

        # Setup tracer
        if collect_trace:
            log_run_begin = partial(tracer.log,
                                    self.exec_timer_name + "-ins-run-begin")
            log_run_end = partial(tracer.log,
                                  self.exec_timer_name + "-ins-run-end")
        else:

            def log_run_begin(*_, **__):
                pass

            log_run_end = log_run_begin

        # Execute
        timers(self.exec_timer_name).start(sync_func=sync_func)

        for instruction in self.instructions:
            #self.worker.sync()
            #print(f"memory_allocated: "
            #      f"{self.worker.get_memory_allocated()/1024**3:.3f} GB  "
            #      f"max_memory_allocated: "
            #      f"{self.worker.get_max_memory_allocated()/1024**3:.3f} GB "
            #      f"next instruction: {instruction}", flush=True)

            if instruction.opcode == PipelineInstType.RUN:
                log_run_begin(instruction.info, sync_func=sync_func)
                self.worker.run_executable(instruction.task_uuid,
                                           instruction.input_uuids,
                                           instruction.output_uuids,
                                           **instruction.opaques["kwargs"])
                log_run_end(instruction.info, sync_func=sync_func)
            elif instruction.opcode == PipelineInstType.SEND:
                self.worker.run_resharding_send_task(instruction.task_uuid,
                                                     instruction.input_uuids[0])
            elif instruction.opcode == PipelineInstType.RECV:
                self.worker.run_resharding_recv_task(
                    instruction.task_uuid, instruction.output_uuids[0],
                    instruction.opaques["set_empty_buffer"])
                # TODO(lmzheng): move this to run_resharding_recv_task
                if instruction.opaques["allgather_uuid"] is not None:
                    task_uuid = instruction.opaques["allgather_uuid"]
                    ary_uuid = instruction.output_uuids[0]
                    self.worker.run_executable(task_uuid, [ary_uuid],
                                               [ary_uuid], False, False)
            elif instruction.opcode == PipelineInstType.BROADCAST:
                self.worker.run_resharding_broadcast_task(
                    instruction.task_uuid,
                    (instruction.input_uuids if instruction.input_uuids
                     is not None else instruction.output_uuids)[0])
            elif instruction.opcode == PipelineInstType.FREE:
                self.worker.delete_buffers(instruction.input_uuids)

        timers(self.exec_timer_name).stop(sync_func=sync_func)

        # copy to global env
        assert len(self.output_local_uuids) == len(output_global_uuids)
        for local_id, global_id in zip(self.output_local_uuids,
                                       output_global_uuids):
            self.global_buffers[global_id] = buffers[local_id]
        # restore global environment
        self.worker.buffers = self.global_buffers
        buffers.clear()
        if global_config.enable_overlapping:
            xe.reset_event_context(self.worker.backend)

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


def dump_stage_execution_trace_internal(stage_execution_info, filename: str):
    """Dump stage execution info as a chrome tracing file."""

    def get_color(i):
        color_list = [
            "thread_state_uninterruptible",
            "thread_state_iowait",
            "thread_state_running",
            "thread_state_runnable",
            "thread_state_unknown",
            "background_memory_dump",
            "light_memory_dump",
            "detailed_memory_dump",
            "vsync_highlight_color",
            "generic_work",
            "good",
            "bad",
            "terrible",
            "yellow",
            "olive",
            "rail_response",
            "rail_animation",
            "rail_idle",
            "rail_load",
            "startup",
            "heap_dump_stack_frame",
            "heap_dump_object_type",
            "heap_dump_child_node_arrow",
            "cq_build_running",
            "cq_build_passed",
            "cq_build_failed",
            "cq_build_attempt_runnig",
            "cq_build_attempt_passed",
            "cq_build_attempt_failed",
        ]
        return color_list[i % len(color_list)]

    slot_list = []
    for request_id, request_timeline in enumerate(zip(*stage_execution_info)):
        sorted_timeline = sorted(request_timeline, key=lambda x: x[0])

        for stage_num, (s, e, node_ids, devices) in enumerate(sorted_timeline):
            for node_id, devices_per_node in zip(node_ids, devices):
                for device_id in devices_per_node:
                    slot = {
                        "name": f"r{request_id}s{stage_num}",
                        "cat": f"request {request_id}, stage {stage_num}",
                        "ph": "X",
                        "pid": int(node_id),
                        "tid": int(device_id),
                        "ts": float(s) * 1e6,
                        "dur": float(e - s) * 1e6,
                        "cname": get_color(request_id)
                    }
                    slot_list.append(slot)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as fout:
        fout.write(
            json.dumps({
                "traceEvents": slot_list,
                "displayTimeUnit": "ms",
            }))
