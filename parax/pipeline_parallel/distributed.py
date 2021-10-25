from collections import namedtuple
from dataclasses import dataclass
import enum
from typing import Any, Dict, Sequence, Set

import numpy as np
from jax.core import Var
from jax.interpreters import pxla

from parax.device_mesh import MeshHostWorker, PhysicalDeviceMesh
from parax.mesh_executable import (AllocZeroBufferWorkerExecutable,
                                   PartialGradAccMeshWorkerExecutable,
                                   get_grad_sync_channel_ids_with_hint,
                                   next_mesh_executable_uuid)
from parax.pipeline_parallel.cross_mesh_resharding import ReshardingTask
from parax.pipeline_parallel.schedules import GpipeSchedule
from parax.pipeline_parallel.stage import XlaShardedPipelineStage
from util import OrderedSet, get_shard_shape


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


AllocateZeroWorkerExecutableConfig = namedtuple(
    "AllocZeroWorkerExecutableConfig",
    ["exec_uuid", "grad_shard_shapes", "grad_shard_dtypes"])
PartialGradWorkerExecutableConfig = namedtuple(
    "GradAccWorkerExecutableConfig",
    ["exec_uuid", "hlo_proto", "strategy_config", "grad_sync_channel_ids"])


def create_task_configs(schedule: GpipeSchedule,
                        stages: Sequence[XlaShardedPipelineStage],
                        meshes: Sequence[PhysicalDeviceMesh],
                        grad_dummy_invars: Set[Var], num_batch, get_next_uuids,
                        instruction_lists, var_at):
    """
    Assign uuids for each task and prepare configs,
    as a replacement of MeshWorkerExecutable.__init__
    """
    num_mesh = len(meshes)
    executable_config_lists = dict()
    resharding_config_lists = dict()

    executable_uuids = []

    # Each worker has its own instruction list because Resharding is not SPMD
    for physical_mesh in meshes:
        for worker in physical_mesh.workers:
            instruction_lists[worker] = list()
            executable_config_lists[worker] = list()
            resharding_config_lists[worker] = list()

    # 1. AllocZeroBuffer executables
    mesh_grad_vars = [dict() for _ in range(num_mesh)]
    # TODO(yonghao): replicated code. abstract this part?
    # collect buffers to allocate in each mesh
    for stage_idx, stage in enumerate(stages):
        mesh_indices = list(schedule.stage_placement(stage_idx))
        assert len(mesh_indices) == 1
        mesh_idx = mesh_indices[0]
        grad_var_spec_dict = mesh_grad_vars[mesh_idx]
        input_specs = stage.input_sharding_specs
        for var_idx, invar in enumerate(stage.invars):
            if invar in grad_dummy_invars:
                if invar in grad_var_spec_dict:
                    raise NotImplemented(
                        f'accumulate {invar} in a mesh but multiple stages')
                grad_var_spec_dict[invar] = input_specs[var_idx]
    if len(grad_var_spec_dict):
        for mesh_idx in range(num_mesh):
            grad_var_spec_dict = mesh_grad_vars[mesh_idx]
            grad_vars, grad_sharding_specs = list(
                zip(*grad_var_spec_dict.items()))
            grad_avals = [var.aval for var in grad_vars]
            grad_shard_shapes = [
                get_shard_shape(aval, spec)
                for aval, spec in zip(grad_avals, grad_sharding_specs)
            ]
            grad_shard_dtypes = [aval.dtype for aval in grad_avals]

            physical_mesh = meshes[mesh_idx]
            exec_uuid = next_mesh_executable_uuid()
            output_uuids = get_next_uuids(
                len(grad_vars) * physical_mesh.total_devices).reshape(
                    len(physical_mesh.workers), len(grad_vars), -1)
            for worker_idx, worker in enumerate(physical_mesh.workers):
                executable_config_lists[worker].append(
                    AllocateZeroWorkerExecutableConfig(exec_uuid,
                                                       grad_shard_shapes,
                                                       grad_shard_dtypes))
                instruction_lists[worker].append(
                    PipelineInstruction.RUN(exec_uuid, [],
                                            output_uuids[worker_idx], {}))
            # (args, workers, devices)
            output_uuids = output_uuids.transpose([1, 0, 2])
            for var_idx, var in enumerate(grad_vars):
                # TODO(yonghao): only works for GPipeSchedule
                key = (repr(var), num_batch - 1)
                var_at.setdefault(key, dict())[mesh_idx] = output_uuids[var_idx]
    # 2. PartialGradAccMeshExecutable
    for stage_idx, stage in enumerate(stages):
        exec_uuid = next_mesh_executable_uuid()
        executable_uuids.append(exec_uuid)

        mesh_idx = schedule.stage_worker_mapping[stage_idx]
        compiled = stage.get_compiled()
        hlo_module = compiled.hlo_modules()[0]
        hlo_proto = hlo_module.as_serialized_hlo_module_proto()
        strategy_config = stage.strategy_config
        grad_sync_channel_ids = get_grad_sync_channel_ids_with_hint(
            hlo_module, stage.output_acc_grad_indices)
        for worker in meshes[mesh_idx].workers:
            executable_config_lists[worker].append(
                PartialGradWorkerExecutableConfig(exec_uuid, hlo_proto,
                                                  strategy_config,
                                                  grad_sync_channel_ids))
    return executable_config_lists, executable_uuids


def split_input_to_microbatches(global_invars, not_batch_invars,
                                donated_invar_set, var_at, num_batch,
                                meshes: Sequence[PhysicalDeviceMesh], stages,
                                schedule, get_next_uuids):
    """
    Returns:
    mesh_arg_indices (Sequence[Sequence[int]]):
        indices[mesh_idx][i] indicates the index of global_invars(expanded) of the i-th input for PipelineWorkerExecutable
        in mesh_idx-th mesh
    donated_invar_list (Sequence[Sequence[bool]]):
        list[mesh_idx] is the donate_invars of PipelineWorkerExecutable
        in mesh_idx-th mesh
    input_indices_list (Sequence[Sequence[Tuple[Index, ...]]]):
        list[mesh_idx] is the input_indices of PipelineWorkerExecutable
        in mesh_idx-th mesh. Here the input_indices are for XLA
        to shard_args instead of indices in input list
    input_local_uuid_list (Dict[MeshHostWorker, np.ndarray]):

    """
    num_mesh = len(meshes)
    global_invar_indices = dict()
    invar_counter = 0
    mesh_arg_lists = [None for _ in range(num_mesh)]
    # expand barch args
    for invar in global_invars:
        if invar in not_batch_invars:
            key = invar, 0
            global_invar_indices[key] = invar_counter
            invar_counter += 1
            continue
        for batch_idx in range(num_batch):
            key = invar, batch_idx
            global_invar_indices[key] = invar_counter
            invar_counter += 1
    # dispatch args to each mesh
    for mesh_idx in range(num_mesh):
        mesh_arg_set = OrderedSet()
        var_to_spec = dict()
        for stage_idx in schedule.worker_stage_mapping[mesh_idx]:
            stage = stages[stage_idx]
            for spec, invar in zip(stage.input_sharding_spec, stage.invars):
                if invar in global_invars:
                    var_to_spec[invar] = spec
                    if invar in not_batch_invars:
                        mesh_arg_set.add((invar, 0))
                        continue
                    for batch_idx in range(num_batch):
                        mesh_arg_set.add((invar, batch_idx))
        mesh_arg_lists[mesh_idx] = list(mesh_arg_set)
    # get local uuids for each input:
    input_local_uuid_list = dict()
    for mesh_idx, physical_mesh in enumerate(meshes):
        mesh_arg_list = mesh_arg_lists[mesh_idx]
        num_args = len(mesh_arg_list)
        # (num_args, num_hosts, num_device)
        arg_uuids = get_next_uuids(
            num_args * physical_mesh.total_devices).reshape(
                num_args, -1, physical_mesh.num_devices_per_host)
        for arg_idx, key in enumerate(mesh_arg_lists[mesh_idx]):
            key = repr(key[0]), key[1]
            var_at.setdefault(key, dict())[mesh_idx] = arg_uuids[arg_idx]
            for worker_idx, worker in enumerate(physical_mesh.workers):
                input_local_uuid_list[worker].append(arg_uuids[arg_idx,
                                                               worker_idx])
    # get returned value
    donated_invar_list = [[
        key[0] in donated_invar_set for key in mesh_arg_list
    ] for mesh_arg_list in mesh_arg_lists]
    mesh_arg_indices = [[global_invar_indices[key]
                         for key in mesh_arg_list]
                        for mesh_arg_list in mesh_arg_lists]
    input_indices_list = [[
        pxla.spec_to_indices(key[0].aval, var_to_spec[key[0]])
        for key in mesh_arg_list
    ]
                          for mesh_arg_list in mesh_arg_lists]
    return (mesh_arg_indices, donated_invar_list, input_indices_list,
            input_local_uuid_list)


def collect_output_from_meshes(global_outvars, var_at, meshes, schedule,
                               stages):
    """
    output_local_uuid_list (Dict[MeshHostWorker, Sequence[np.ndarray]]):
        output local uuid of each MeshHostWorker
    mesh_output_indices_list (Sequence[Dict[int, int]]):
        list[outvar_idx][mesh_idx] indicates the index of the output in
        that mesh corresponding to outvar_idx-th global outputs
    output_spec_list (Sequence[Sequence[ShardingSpec]]):
        list[mesh_idx] is the ShardingSpec of all outputs from 
        PipelineWorkerExecutable in mesh_idx-th mesh.
    """
    output_local_uuid_list = dict()
    num_mesh = len(meshes)

    for mesh in meshes:
        for worker in mesh.workers:
            output_local_uuid_list[worker] = []
    mesh_output_indices_list = []
    output_spec_list = [[] for _ in range(num_mesh)]
    # collect outvar specs
    var_to_spec_all_meshes = []
    global_outvar_set = set(global_outvars)
    for mesh_idx in range(num_mesh):
        var_to_spec = dict()
        for stage_idx in schedule.worker_stage_mapping[mesh_idx]:
            stage = stages[stage_idx]
            for spec, outvar in zip(stage.output_sharding_spec, stage.outvars):
                if outvar in global_outvar_set:
                    var_to_spec[outvar] = spec
        var_to_spec_all_meshes.append(var_to_spec)
    # assign indices and get specs
    for outvar in global_outvars:
        # the apply gradient only writes to microbatch 0
        key = (repr(outvar), 0)
        var_meshes = var_at[key]
        mesh_out_indices = dict()
        for mesh_idx in var_meshes:
            mesh = meshes[mesh_idx]
            uuids = var_meshes[mesh_idx]
            for worker_idx, worker in enumerate(mesh.workers):
                output_local_uuid_list[worker].append(uuids[worker_idx])
            mesh_out_indices[mesh_idx] = (len(output_local_uuid_list[worker]) -
                                          1)
            output_spec_list[mesh_idx].append(
                var_to_spec_all_meshes[mesh_idx][outvar])
        mesh_output_indices_list.append(mesh_out_indices)
    return mesh_output_indices_list, output_local_uuid_list, output_spec_list


def flatten_uuid_set(container):
    # From Sequence[np.ndarray] to set of elements in the array
    container = list(container)
    output = set()
    for e in container:
        if isinstance(e, int):
            output.add(e)
            continue
        output.union(list(e))
    return output


# TODO: This function requires prepared resharding tasks as inputs, which relies on sharding specs of each stage.
# However, the function is called before we set runnables, so the order new should be modified: call get_compiled first for sharding specs, then create communicator.
# After that, run this function and finally launch instructions.

# The input of a PipelineWorkerExecutable has multiple minibatches.
# We define their order as:
# invar0, invar1, ... invarX_mb0, invarX_mb1, ... invarX+1, ...
# That is, following the order of global_invars. If it is a batch var,
# we then expand it.
# Shall we pass this to runtime? or just follow this rule everywhere is ok?


# There are too many inputs and outputs, why not merge this function into
# the new runtime and set its member directly.
def create_instructions_from_pipeline_schedule(
        schedule: GpipeSchedule, stages: Sequence[XlaShardedPipelineStage],
        resharding_tasks, meshes: Sequence[PhysicalDeviceMesh],
        grad_dummy_invars: Set[Var], global_invars, global_outvars, is_batch,
        num_batch, donated_invar_set):
    """
    This function allocates uuids of intermediates,
    as well as creating instruction lists for all intermediates
    """
    uuid_counter = 0

    def get_next_uuids(num) -> np.ndarray:
        nonlocal uuid_counter
        ret = np.arange(start=uuid_counter,
                        stop=uuid_counter + num,
                        dtype=np.int64)
        uuid_counter += num
        return ret

    def get_invar_key(invar, batch_idx):
        if invar in not_batch_invars:
            var_key = repr(invar)
            key = (repr(invar), 0)
        # TODO(yonghao): only works for GPipeSchedule, move this fn there?
        elif (invar in grad_dummy_invars and batch_idx < num_batch - 1):
            var_key = grad_dummy_invars[invar]
            key = (var_key, batch_idx + 1)
        else:
            var_key = repr(invar)
            key = (repr(invar), batch_idx)
        return var_key, key

    num_mesh = len(meshes)
    instruction_lists = dict()
    var_at = dict()  # var->(mesh_idx->uuids in shape of (worker, deice))

    # Microbatch-unrelated work
    # compile args for tasks
    (executable_config_lists, executable_uuids) = create_task_configs(
        schedule, stages, meshes, grad_dummy_invars, num_batch, get_next_uuids,
        instruction_lists, var_at)
    # mesh_arg_indices
    not_batch_invars = set(
        [var for var, batch in zip(global_invars, is_batch) if not batch])
    (mesh_arg_indices, donated_invar_list, input_indices_list,
     input_local_uuid_list) = split_input_to_microbatches(
         global_invars, not_batch_invars, donated_invar_set, var_at, num_batch,
         meshes, stages, schedule, get_next_uuids)

    # Microbatch-related work
    for _, sched in enumerate(schedule.schedules):
        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            physical_mesh = meshes[mesh_idx]
            num_devices_per_host = physical_mesh.num_devices_per_host
            batch_idx, stage_idx = task
            stage = stages[stage_idx]
            received_keys = OrderedSet()
            # shard_args for intermediates
            for invar in stage.invars:
                var_key, key = get_invar_key(invar, batch_idx)
                if mesh_idx in var_at[key]:
                    # have a copy at the current mesh
                    continue
                if len(var_at[key]) > 1:
                    raise NotImplemented("Not support resharding replicated")

                src_idx, src_uuids = list(var_at[key].items())[0]
                send_buf_uuids = {
                    worker: list() for worker in meshes[src_idx].workers
                }
                recv_buf_uuids = {
                    worker: list() for worker in physical_mesh.workers
                }
                num_sender_host = len(send_buf_uuids)
                num_receiver_host = len(recv_buf_uuids)

                resharding_task = resharding_tasks[src_idx][mesh_idx][var_key]
                resharding_task: ReshardingTask

                # collect uuids of each send_tile in each worker according to resharding_task's plan
                for sender_str in resharding_task.sender_uuid_plan:
                    send_worker = resharding_task.collective_group.device_str_to_mesh_worker_map[
                        sender_str]
                    send_buf_flat_idx = resharding_task.task_spec.src.device_str_to_flat_index[
                        sender_str]
                    send_buf_host = send_buf_flat_idx // num_sender_host
                    send_buf_device = send_buf_flat_idx % num_sender_host
                    send_buf_uuids[send_worker].append(
                        src_uuids[send_buf_host, send_buf_device])

                # add send tasks for each worker
                for w, task_uuid in resharding_task.send_worker_task_ids.items(
                ):
                    input_uuids = send_buf_uuids[w]
                    instruction_lists[w].append(
                        PipelineInstruction.SEND(task_uuid, input_uuids))

                recv_uuids = get_next_uuids(
                    physical_mesh.total_devices).reshape(
                        num_receiver_host, num_devices_per_host)
                received_keys.add(key)
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
                for w, task_uuid in resharding_task.recv_worker_task_ids.items(
                ):
                    output_uuids = recv_buf_uuids[w]
                    instruction_lists[w].append(
                        PipelineInstruction.RECV(task_uuid, output_uuids, True))
                # no need to set default because of sent mesh
                var_at[key][mesh_idx] = recv_uuids
            # execute
            # allocate uuids for buffers created by RUN
            for outvar in stage.outvars:
                key = (repr(outvar), batch_idx)
                # get uuids of this outvar
                var_at.setdefault(key,
                                  default=dict())[mesh_idx] = get_next_uuids(
                                      physical_mesh.total_devices).reshape(
                                          num_receiver_host,
                                          num_devices_per_host)

            exec_uuid = executable_uuids[stage_idx]
            for worker_idx, worker in enumerate(physical_mesh.workers):
                # Get input and output uuids. They should be at the mesh
                input_uuids = np.zeros(
                    (len(stage.invars), num_devices_per_host))
                output_uuids = np.zeros(
                    (len(stage.outvars), num_devices_per_host))
                for idx, invar in enumerate(stage.invars):
                    _, key = get_invar_key(invar, batch_idx)
                    input_uuids[idx] = var_at[key][mesh_idx][worker_idx]
                for idx, outvar in enumerate(stage.outvars):
                    key = (repr(outvar, batch_idx))
                    output_uuids[idx] = var_at[key][mesh_idx][worker_idx]

                # TODO(yonghao): only works for GPipeSchedule.
                kwargs = {
                    "skip_grad_sync": not (stage_idx >= num_mesh and
                                           stage_idx < num_mesh * 2 and
                                           batch_idx == 0)
                }

                instruction_lists[worker].append(
                    PipelineInstruction.RUN(exec_uuid, input_uuids,
                                            output_uuids, kwargs))
            # free all received buffers
            received_uuids = [
                var_at[key].pop(mesh_idx) for key in received_keys
            ]
            for worker_idx, worker in enumerate(physical_mesh.workers):
                instructions = instruction_lists[worker]
                for uuids in received_uuids:
                    instructions.append(
                        PipelineInstruction.FREE(uuids[worker_idx]))
    # output info
    (mesh_output_indices_list, output_local_uuid_list,
     output_spec_list) = collect_output_from_meshes(global_outvars, var_at,
                                                    meshes)
    # add FREE insts
    for worker in instruction_lists:
        instruction_list: Sequence[PipelineInstruction] = instruction_lists[
            worker]
        new_list = []
        used_later_uuids = flatten_uuid_set(output_local_uuid_list[worker])
        for instruction in reversed(instruction_list):
            input_uuids = flatten_uuid_set(list(instruction.input_uuids))
            # for free instruction, do not free again
            if instruction.opcode != PipelineInstType.FREE:
                unused_uuids = list(input_uuids.difference(used_later_uuids))
                new_list.append(PipelineInstruction.FREE(
                    np.array(unused_uuids)))
            new_list.append(instruction)
            used_later_uuids.update(input_uuids)

    return (instruction_lists, executable_config_lists, mesh_arg_indices,
            donated_invar_list, input_indices_list, input_local_uuid_list,
            mesh_output_indices_list, output_local_uuid_list, output_spec_list)


# TODO(Hao):
#  1. implement a base class, let two runtimes inherit this baseclass
#  2. make some abstraction, put sharing-related functionality in baseclass, such as:
#  creating collective_group, resharding-related logic.
#  3. complete PipelineMeshDriverExecutable main logic.
class PipelineMeshDriverExecutable:

    def __init__(self, physical_meshes):
        self.physical_meshes = physical_meshes
        num_meshes = len(self.physical_meshes)

        # TODO(Hao): only compile, not actually getting runnables, for the sake
        #  of invoking `create_instructions_from_pipeline_schedule`.
        self._compile_xla_stages()

        # TODO(Hao): spawn cgs and communicators, find device-device group relations/dependencies, etc.
        #  Note: do some abstraction and put the functions in baseclass.
        self._create_collective_group()
        # TODO(Hao): for each tensor (that needs to be resharded), create its resharding task_spec/task
        resharding_tasks, communicator, _ = self._prepare_communicator()

        # call the main compile function to get instructions.
        instruction_lists, executable_config_lists, resharding_config_lists = \
            create_instructions_from_pipeline_schedule(
            schedule, # generated in three_d_parallel.py
            stages, # generated in three_d_parallel, but we have compiled it for communicator construciton,
            resharding_tasks, # generated above
            meshes, # physical, generated outside
        )

        # TODO(Hao): for each worker,
        #  spawn many PipelineMeshWorkerExecutable remotely
        #  Then, send the compiled instructons + configs etc. using the class `PipelineMeshWorkerExecutable`

        # Done (Yonghao): compiled output, set here.
        #  do it together with global input/output uuid-buffer mapping.
        self.mesh_arg_indices = [None for _ in range(num_meshes)]
        self.donate_invars = [None for _ in range(num_meshes)]
        self.input_indices = [None for _ in range(num_meshes)]

        # TODO (Hao): where to put RDA?

    def run(self, *args):
        """1. shard inputs (send input to each remote worker)
            2. launch each worker to start instructions,
            sync after each iter"""
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
                 output_local_uuids: Sequence[int], executable_configs,
                 donate_invars, worker: MeshHostWorker):
        # Instruction Lists
        self.instructions = instructions
        self.input_local_uuids = input_local_uuids
        self.output_local_uuids = output_local_uuids
        self.donate_invars = donate_invars
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
        for task_config in executable_configs:
            if isinstance(task_config, PartialGradWorkerExecutableConfig):
                self.worker.put_executable(task_config.exec_uuid,
                                           PartialGradAccMeshWorkerExecutable,
                                           task_config.hlo_proto,
                                           task_config.strategy_config,
                                           task_config.grad_sync_channel_ids)
                continue
            assert isinstance(task_config, AllocateZeroWorkerExecutableConfig)
            self.worker.put_executable(task_config.exec_uuid,
                                       AllocZeroBufferWorkerExecutable,
                                       task_config.grad_shard_shapes,
                                       task_config.grad_shard_dtypes)

    def execute_on_worker(self, input_global_uuids, output_global_uuids):
        # copy to local env
        assert len(self.input_local_uuids) == len(input_global_uuids)
        buffers = dict()
        for local_ids, global_ids in zip(self.input_local_uuids,
                                         input_global_uuids):
            local_ids = list(local_ids)
            global_ids = list(global_ids)
            for local_id, global_id in zip(local_ids, global_ids):
                buffers[local_id] = self.global_buffers[global_id]
        # donate invars
        for global_ids, donate in zip(input_global_uuids, self.donate_invars):
            if donate:
                self.worker.delete_buffers(list(global_ids))
        # monkey patch
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

        # copy to global env
        assert len(self.output_local_uuids) == len(output_global_uuids)
        for local_ids, global_ids in zip(self.output_local_uuids,
                                         output_global_uuids):
            local_ids = list(local_ids)
            global_ids = list(global_ids)
            for local_id, global_id in zip(local_ids, global_ids):
                self.global_buffers[global_id] = buffers[local_id]

        # monkey patch
        self.worker.buffers = self.global_buffers
        # Clean the dict
        buffers.clear()
        return True