from collections import namedtuple
from dataclasses import dataclass
import enum
from typing import Any, Dict, Sequence

import numpy as np

from parax.parax.device_mesh import AbstractMeshWorker, MeshHostWorker
from parax.parax.mesh_executable import PartialGradAccMeshWorkerExecutable
from parax.parax.pipeline_parallel.cross_mesh_resharding import ReshardingTask
from parax.parax.pipeline_parallel.runtime import GpipeSchedule
from parax.parax.pipeline_parallel.stage import XlaShardedPipelineStage
from parax.parax.util import xla_buffer_to_jax_buffer

BufferRefType = namedtuple("", ["is_static", "value"])


class PipelineInstType(enum.IntEnum):
    RUN = 0
    SEND = 1
    RECV = 2
    FREE = 3


@dataclass
class PipelineInstruction:
    opcode: PipelineInstType
    task_uuid: int
    input_refs: np.ndarray
    output_refs: np.ndarray
    opaques: Dict[str, Any]

    @classmethod
    def RUN(cls, task_uuid, input_refs, output_refs, kwargs):
        return cls(opcode=PipelineInstType.RUN,
                   task_uuid=task_uuid,
                   input_refs=input_refs,
                   output_refs=output_refs,
                   opaques={'kwargs': kwargs})

    @classmethod
    def SEND(cls, task_uuid, input_refs):
        return cls(opcode=PipelineInstType.SEND,
                   task_uuid=task_uuid,
                   input_refs=input_refs,
                   output_refs=None,
                   opaques=None)

    @classmethod
    def RECV(cls, task_uuid, output_refs, set_empty_buffer):
        return cls(opcode=PipelineInstType.RECV,
                   task_uuid=task_uuid,
                   input_refs=None,
                   output_refs=output_refs,
                   opaque={'set_empty_buffer': set_empty_buffer})

    @classmethod
    def FREE(cls, input_refs):
        return cls(opcode=PipelineInstType.FREE,
                   task_uuid=None,
                   input_refs=input_refs,
                   output_refs=None,
                   opaque=None)


MeshWorkerExecutableConfig = namedtuple(
    "WorkerExecutableConfig",
    ["exec_uuid", "hlo_proto", "strategy_config", "grad_sync_channel_ids"])
ReshardingTaskConfig = namedtuple("ReshardingTaskConfig",
                                  ["task_type", "uuid", "tasks", "group_name"])


# TODO: This function requires prepared resharding tasks as inputs, which relies on sharding specs of each stage.
# However, the function is called before we set runnables, so the order new should be modified: call get_compiled first for sharding specs, then create communicator.
# After that, run this function and finally launch instructions.
def create_instructions_from_pipeline_schedule(
        schedule: GpipeSchedule, stages: Sequence[XlaShardedPipelineStage],
        resharding_tasks):
    """
    This function allocates refs of intermediates, as well as creating instruction lists for all intermediates
    """
    ref_counter = 0

    def get_next_refs(num):
        nonlocal ref_counter
        ret = np.asarray(range(ref_counter, ref_counter + num), dtype=np.int64)
        ref_counter += num
        return ret

    num_meshes = schedule.meshes
    pipeline_instruction_lists = [[] for _ in range(num_meshes)]
    var_at = dict()  # Var -> Set[int](set of mesh indices)

    for _, sched in enumerate(schedule.schedules):
        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            batch_idx, stage_idx = task
            instructions = pipeline_instruction_lists[mesh_idx]
            stage = stages[stage_idx]
            for invar in stage.invars:
                key = (repr(invar), batch_idx)
                if key not in var_at:
                    # global variable, always sharded ready
                    continue
                if mesh_idx in var_at[key]:
                    # have a copy at the current mesh
                    continue
                if len(var_at[key]) > 1:
                    raise NotImplemented("Not support resharding replicated")
                src_mesh_idx = list(var_at[key])[0]
                resharding_task: ReshardingTask = resharding_tasks[
                    src_mesh_idx][mesh_idx][key]

                for worker, in resharding_task.send_worker_tasks.items():
                    pass

                # TODO: add send tasks for each worker
                pipeline_instruction_lists[src_mesh_idx].append(
                    PipelineInstruction.SEND())
                # TODO: allocate refs for buffers created by RECV
                instructions.append(PipelineInstruction.RECV())
            # TODO: allocate refs for buffers created by RUN
            # TODO: get input buffers
            instructions.append(PipelineInstruction.RUN())
            for outvar in stage.outvars:
                key = (repr(outvar), batch_idx)
                # TODO: instead of a set, use a dict as: mesh_idx -> uuid for each device
                var_at.setdefault(key, default=set()).add(mesh_idx)

    return pipeline_instruction_lists


# TODO: add this into Jax3DPipeline
class PipelineMeshDriverExecutable:

    def __init__(self, physical_meshes):
        self.physical_meshes = physical_meshes
        num_meshes = len(self.physical_meshes)
        # TODO: Create tasks
        # TODO: Create task_configs
        # TODO
        self.mesh_arg_indices = [None for _ in range(num_meshes)]
        self.donate_invars = [None for _ in range(num_meshes)]
        self.input_indices = [None for _ in range(num_meshes)]

    def launch_on_driver(self, *args):
        num_meshes = len(self.physical_meshes)
        input_bufs = [None for _ in range(num_meshes)]
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            mesh_args = [args[idx] for idx in self.mesh_arg_indices[mesh_idx]]
            input_bufs[mesh_idx] = physical_mesh.shard_args(
                self.input_indices[mesh_idx], self.donate_invars[mesh_idx],
                mesh_args)


class PipelineMeshWorkerExecutable(AbstractMeshWorker):

    def __init__(self, instructions: Sequence[PipelineInstruction],
                 input_local_refs: Sequence[int],
                 output_local_refs: Sequence[int],
                 executable_configs: Sequence[MeshWorkerExecutableConfig],
                 resharding_configs: Sequence[ReshardingTaskConfig],
                 worker: MeshHostWorker):
        super(PipelineMeshWorkerExecutable, self).__init__(worker.backend)
        # Instruction Lists
        self.instructions = instructions
        self.input_local_refs = input_local_refs
        self.output_local_refs = output_local_refs
        # Buffer management
        self.worker = worker
        self.global_buffers = worker.buffers
        # Create tasks
        for task_config in executable_configs:
            self.put_executable(task_config.exec_uuid,
                                PartialGradAccMeshWorkerExecutable,
                                task_config.hlo_proto,
                                task_config.strategy_config,
                                task_config.grad_sync_channel_ids)

        for task_config in resharding_configs:
            if task_config.task_type == "send":
                self.put_resharding_send_task(task_config.uuid,
                                              task_config.tasks,
                                              task_config.group_name)
                continue
            self.put_resharding_recv_task(task_config.uuid, task_config.tasks,
                                          task_config.group_name)

    def execute_on_worker(self, input_global_uuids, output_global_uuids):
        assert len(self.input_local_refs) == len(input_global_uuids)
        for local_id, global_id in zip(self.input_local_refs,
                                       input_global_uuids):
            self.buffers[local_id] = self.global_buffers[global_id]
        # Execute
        for instruction in self.instructions:
            if instruction.opcode == PipelineInstType.RUN:
                self.run_executable(instruction.task_uuid,
                                    instruction.input_refs,
                                    instruction.output_refs,
                                    **instruction.opaques["kwargs"])
            elif instruction.opcode == PipelineInstType.SEND:
                self.run_resharding_send_task(instruction.task_uuid,
                                              instruction.input_refs)
            elif instruction.opcode == PipelineInstType.RECV:
                self.run_resharding_recv_task(
                    instruction.task_uuid, instruction.output_refs,
                    instruction.opaques['set_empty_buffer'])
            elif instruction.opcode == PipelineInstType.FREE:
                self.delete_buffers(instruction.input_refs)

        for local_id, global_id in zip(self.output_local_refs,
                                       output_global_uuids):
            self.global_buffers[global_id] = self.buffers[local_id]
        # Clean the dict
        return True

    # The Executable has no ability to do Cross Mesh Resharding,
    # instead, it uses corresponding worker's impl
    def send_tile(self, uuid, offset, dst_rank, dst_gpu_idx, group_name):
        src_buffer = xla_buffer_to_jax_buffer(self.buffers[uuid])
        self.worker.send_tile_impl(src_buffer, offset, dst_rank, dst_gpu_idx,
                                   group_name)
        return True

    def recv_tile(self, uuid, device_id, indices_in_dst_tile, src_rank,
                  src_gpu_idx, group_name):
        if uuid not in self.buffers:
            raise RuntimeError()
        self.buffers[uuid] = self.worker.recv_tile_impl(self.buffers[uuid], device_id,
                                                 indices_in_dst_tile, src_rank,
                                                 src_gpu_idx, group_name)
        return True

    def sync(self):
        self.worker.sync()

    def shutdown(self):
        self.sync()
        del self.buffers
        del self.executables