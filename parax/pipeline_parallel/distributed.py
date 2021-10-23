from collections import namedtuple
from dataclasses import dataclass
import enum
from typing import Any, Dict, Sequence

import numpy as np

from parax.device_mesh import AbstractMeshWorker, MeshHostWorker, PhysicalDeviceMesh
from parax.mesh_executable import PartialGradAccMeshWorkerExecutable, next_mesh_executable_uuid
from parax.pipeline_parallel.cross_mesh_resharding import ReshardingTask, next_resharding_task_uuid
from parax.pipeline_parallel.runtime import GpipeSchedule
from parax.pipeline_parallel.stage import XlaShardedPipelineStage


class PipelineInstType(enum.IntEnum):
    RUN = 0
    SEND = 1
    RECV = 2
    FREE = 3


@dataclass
class PipelineInstruction:
    opcode: PipelineInstType
    task_uuid: int
    input_uuids: np.ndarray
    output_uuids: np.ndarray
    opaques: Dict[str, Any]

    @classmethod
    def RUN(cls, task_uuid, input_uuids, output_uuids, kwargs):
        return cls(opcode=PipelineInstType.RUN,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=output_uuids,
                   opaques={'kwargs': kwargs})

    @classmethod
    def SEND(cls, task_uuid, input_uuids):
        return cls(opcode=PipelineInstType.SEND,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=None,
                   opaques=None)

    @classmethod
    def RECV(cls, task_uuid, output_uuids, set_empty_buffer):
        return cls(opcode=PipelineInstType.RECV,
                   task_uuid=task_uuid,
                   input_uuids=None,
                   output_uuids=output_uuids,
                   opaque={'set_empty_buffer': set_empty_buffer})

    @classmethod
    def FREE(cls, input_uuids):
        return cls(opcode=PipelineInstType.FREE,
                   task_uuid=None,
                   input_uuids=input_uuids,
                   output_uuids=None,
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
        resharding_tasks, meshes: Sequence[PhysicalDeviceMesh]):
    """
    This function allocates uuids of intermediates, as well as creating instruction lists for all intermediates
    """
    uuid_counter = 0

    def get_next_uuids(num) -> np.ndarray:
        nonlocal uuid_counter
        ret = np.arange(start=uuid_counter,
                        stop=uuid_counter + num,
                        dtype=np.int64)
        uuid_counter += num
        return ret

    num_meshes = len(meshes)
    instruction_lists = dict()
    executable_config_lists = dict()
    resharding_config_lists = dict()
    var_at = dict(
    )  # Var -> Dict[int, np.ndarray]: var->(mesh_idx->uuids on the mesh)

    # Each worker has its own instruction list because Resharding is not SPMD
    for physical_mesh in meshes:
        for worker in physical_mesh.workers:
            instruction_lists[worker] = list()
            executable_config_lists[worker] = list()
            resharding_config_lists[worker] = list()

    for _, sched in enumerate(schedule.schedules):
        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            physical_mesh = meshes[mesh_idx]
            num_devices_per_host = physical_mesh.num_devices_per_host
            batch_idx, stage_idx = task
            stage = stages[stage_idx]
            for invar in stage.invars:
                key = (repr(invar), batch_idx)
                if key not in var_at:
                    # global variable, always sharded ready
                    # TODO: assign uuids according to sharding specs
                    continue
                if mesh_idx in var_at[key]:
                    # have a copy at the current mesh
                    continue
                if len(var_at[key]) > 1:
                    raise NotImplemented("Not support resharding replicated")

                src_mesh_idx, src_uuids = list(var_at[key].items())[0]
                send_buf_uuids = {
                    worker: list() for worker in meshes[src_mesh_idx].workers
                }
                recv_buf_uuids = {
                    worker: list() for worker in physical_mesh.workers
                }
                num_sender_host = len(send_buf_uuids)
                num_receiver_host = len(recv_buf_uuids)

                resharding_task: ReshardingTask = resharding_tasks[
                    src_mesh_idx][mesh_idx][key]
                send_tasks, recv_tasks = resharding_task.get_send_recv_tasks()
                group_name = resharding_task.collective_group.group_name

                # collect uuids of each send_tile in each worker according to resharding_task's plan
                for sender_str in resharding_task.sender_uuid_plan:
                    sender_worker = resharding_task.collective_group.device_str_to_mesh_worker_map[
                        sender_str]
                    send_buf_flat_idx = resharding_task.task_spec.src.device_str_to_flat_index[
                        sender_str]
                    send_buf_host = send_buf_flat_idx // num_sender_host
                    send_buf_device = send_buf_flat_idx % num_sender_host
                    send_buf_uuids[sender_worker].append(
                        src_uuids[send_buf_host, send_buf_device])

                # add send tasks for each worker
                for worker, send_task in send_tasks.items():
                    task_uuid = next_resharding_task_uuid()
                    input_uuids = send_buf_uuids[worker]
                    resharding_config_lists[worker].append(
                        ReshardingTaskConfig(task_type="send",
                                             uuid=task_uuid,
                                             tasks=send_task,
                                             group_name=group_name))

                    instruction_lists[worker].append(
                        PipelineInstruction.SEND(task_uuid, input_uuids))

                recv_uuids = get_next_uuids(
                    physical_mesh.total_devices).reshape(
                        num_receiver_host, num_devices_per_host)
                # collect uuids of each recv_tile in each worker according to resharding_task's plan
                for receiver_str in resharding_task.receiver_uuid_plan:
                    receiver_worker = resharding_task.collective_group.device_str_to_host_id_map[
                        receiver_str]
                    recv_buf_flat_idx = resharding_task.task_spec.dst.device_str_to_flat_index[
                        receiver_str]
                    recv_buf_host = recv_buf_flat_idx // num_receiver_host
                    recv_buf_device = recv_buf_flat_idx % num_receiver_host
                    recv_buf_uuids[receiver_worker].append(
                        recv_uuids[recv_buf_host, recv_buf_device])

                # add recv task for each worker
                for worker, recv_task in recv_tasks.items():
                    task_uuid = next_resharding_task_uuid()

                    output_uuids = recv_buf_uuids[worker]

                    resharding_config_lists[worker].append(
                        ReshardingTaskConfig(task_type="recv",
                                             uuid=task_uuid,
                                             task=recv_task,
                                             group_name=group_name))
                    instruction_lists[worker].append(
                        PipelineInstruction.RECV(task_uuid, output_uuids, True))
                var_at[key][mesh_idx] = recv_uuids

            # allocate uuids for buffers created by RUN
            for outvar in stage.outvars:
                key = (repr(outvar), batch_idx)
                # get uuids of this outvar
                var_at.setdefault(key,
                                  default=dict())[mesh_idx] = get_next_uuids(
                                      physical_mesh.total_devices).reshape(
                                          num_receiver_host,
                                          num_devices_per_host)

            exec_uuid = next_mesh_executable_uuid()
            for worker_idx, worker in enumerate(physical_mesh.workers):
                # Get input and output uuids. They should be at the mesh
                input_uuids = np.zeros(
                    (len(stage.invars), num_devices_per_host))
                output_uuids = np.zeros(
                    (len(stage.outvars), num_devices_per_host))
                for idx, invar in enumerate(stage.invars):
                    key = (repr(invar), batch_idx)
                    input_uuids[idx, :] = var_at[key][mesh_idx][worker_idx, :]
                for idx, outvar in enumerate(stage.outvars):
                    key = (repr(outvar, batch_idx))
                    output_uuids[idx, :] = var_at[key][mesh_idx][worker_idx, :]

                # TODO(yonghao): only works for GPipeSchedule.
                kwargs = {
                    "skip_grad_sync": stage_idx > num_meshes / 2
                                      and batch_idx == 0
                }
                # TODO: prepare them here
                hlo_proto = None
                strategy_config = None
                grad_sync_channel_ids = None

                executable_config_lists[worker].append(
                    MeshWorkerExecutableConfig(exec_uuid, hlo_proto,
                                               strategy_config,
                                               grad_sync_channel_ids))

                instruction_lists[worker].append(
                    PipelineInstruction.RUN(exec_uuid, input_uuids,
                                            output_uuids, kwargs))

    return instruction_lists, executable_config_lists, resharding_config_lists


# TODO: merge this into Jax3DPipeline
class PipelineMeshDriverExecutable:

    def __init__(self, physical_meshes):
        self.physical_meshes = physical_meshes
        num_meshes = len(self.physical_meshes)
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


class PipelineMeshWorkerExecutable:

    def __init__(self, instructions: Sequence[PipelineInstruction],
                 input_local_uuids: Sequence[int],
                 output_local_uuids: Sequence[int],
                 executable_configs: Sequence[MeshWorkerExecutableConfig],
                 resharding_configs: Sequence[ReshardingTaskConfig],
                 worker: MeshHostWorker):
        super(PipelineMeshWorkerExecutable, self).__init__(worker.backend)
        # Instruction Lists
        self.instructions = instructions
        self.input_local_uuids = input_local_uuids
        self.output_local_uuids = output_local_uuids
        # Buffer management
        self.worker = worker
        self.global_buffers = worker.buffers
        # Local executables and tasks
        self.executables = dict()
        self.send_tasks = dict()
        self.recv_tasks = dict()
        self.global_executables = worker.executables
        self.global_send_tasks = worker.send_tasks
        self.global_recv_tasks = worker.recv_tasks
        # Create tasks
        self.push_tasks()
        for task_config in executable_configs:
            self.worker.put_executable(task_config.exec_uuid,
                                       PartialGradAccMeshWorkerExecutable,
                                       task_config.hlo_proto,
                                       task_config.strategy_config,
                                       task_config.grad_sync_channel_ids)

        for task_config in resharding_configs:
            if task_config.task_type == "send":
                self.worker.put_resharding_send_task(task_config.uuid,
                                                     task_config.tasks,
                                                     task_config.group_name)
                continue
            self.worker.put_resharding_recv_task(task_config.uuid,
                                                 task_config.tasks,
                                                 task_config.group_name)
        self.pop_tasks()

    def execute_on_worker(self, input_global_uuids, output_global_uuids):
        assert len(self.input_local_uuids) == len(input_global_uuids)
        buffers = dict()
        for local_id, global_id in zip(self.input_local_uuids,
                                       input_global_uuids):
            buffers[local_id] = self.global_buffers[global_id]

        self.push_tasks()
        self.worker.buffers = buffers

        # Execute
        for instruction in self.instructions:
            if instruction.opcode == PipelineInstType.RUN:
                self.worker.run_executable(instruction.task_uuid,
                                           instruction.input_uuids,
                                           instruction.output_uuids,
                                           **instruction.opaques["kwargs"])
            elif instruction.opcode == PipelineInstType.SEND:
                self.worker.run_resharding_send_task(instruction.task_uuid,
                                                     instruction.input_uuids)
            elif instruction.opcode == PipelineInstType.RECV:
                self.worker.run_resharding_recv_task(
                    instruction.task_uuid, instruction.output_uuids,
                    instruction.opaques['set_empty_buffer'])
            elif instruction.opcode == PipelineInstType.FREE:
                self.worker.delete_buffers(instruction.input_uuids)

        for local_id, global_id in zip(self.output_local_uuids,
                                       output_global_uuids):
            self.global_buffers[global_id] = buffers[local_id]

        self.worker.buffers = self.global_buffers
        self.pop_tasks()
        # TODO: Clean the dict
        return True

    def push_tasks(self):
        self.worker.executables = self.executables
        self.worker.send_tasks = self.send_tasks
        self.worker.recv_tasks = self.recv_tasks

    def pop_tasks(self):
        self.worker.executables = self.global_executables
        self.worker.send_tasks = self.global_send_tasks
        self.worker.recv_tasks = self.global_recv_tasks