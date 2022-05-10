"""Decentralized inter-op parallel runtime using static instructions."""
from collections import namedtuple, defaultdict
from dataclasses import dataclass
import enum
import logging
from typing import Any, Dict, Sequence, List, Callable, Optional, Union

from jax.core import Var
from jax.interpreters import pxla
import numpy as np
import ray.exceptions

from alpa.device_mesh import MeshHostWorker, DistributedArray, ReplicatedDistributedArray
from alpa.global_env import global_config
from alpa.mesh_executable import (AllocZeroBufferWorkerExecutable,
                                  MemzeroWorkerExecutable,
                                  PartialGradAccMeshWorkerExecutable,
                                  next_mesh_executable_uuid, get_uuid_np_array,
                                  next_remote_buffer_uuid, RemoteBufferRef)
from alpa.pipeline_parallel.base_runtime import BaseDistributedRuntime
from alpa.pipeline_parallel.cross_mesh_resharding import SymbolicReshardingTask, SymbolicBroadcastReshardingTask
from alpa.pipeline_parallel.schedules import cached_property, PipelineSchedule
from alpa.pipeline_parallel.computation import XlaShardedPipelineComputation
from alpa.pipeline_parallel.device_mesh_group import DistributedPhysicalDeviceMeshGroup
from alpa.timer import timers
from alpa.util import DisjointDict, OrderedSet, get_shard_shape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

timer_names = {
    "overall": "average",
    "compute": "sum",
    "resharding_send": "sum",
    "resharding_recv": "sum",
    "free": "sum",
}


class PipelineInstType(enum.IntEnum):
    """Enum class for pipeline instruction types."""

    # Run an XLA executable
    RUN = 0
    # Run a sending task
    SEND = 1
    # Run a receiving task
    RECV = 2
    # Free tensors
    FREE = 3
    BROADCAST = 4


@dataclass
class PipelineInstruction:
    """Base class for pipeline instructions."""

    opcode: PipelineInstType
    task_uuid: Optional[int]
    input_uuids: Optional[np.ndarray]
    output_uuids: Optional[np.ndarray]
    opaques: Optional[Dict[str, Any]]
    info: str
    print_uuids: bool = False

    @classmethod
    def Run(cls, task_uuid, input_uuids, output_uuids, kwargs, info=""):  # noqa
        return cls(opcode=PipelineInstType.RUN,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=output_uuids,
                   opaques={"kwargs": kwargs},
                   info=info)

    @classmethod
    def Send(cls, task_uuid, input_uuids, info=""):  # noqa
        return cls(opcode=PipelineInstType.SEND,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=None,
                   opaques=None,
                   info=info)

    @classmethod
    def Recv(cls, # noqa
             task_uuid,
             output_uuids,
             set_empty_buffer,
             allgather_uuid=None,
             info=""):  # noqa
        return cls(opcode=PipelineInstType.RECV,
                   task_uuid=task_uuid,
                   input_uuids=None,
                   output_uuids=output_uuids,
                   opaques={
                       "set_empty_buffer": set_empty_buffer,
                       "allgather_uuid": allgather_uuid
                   },
                   info=info)

    @classmethod
    def Broadcast(cls, # noqa
                  task_uuid,
                  input_uuids,
                  output_uuids,
                  info="broadcast"):  # noqa
        return cls(opcode=PipelineInstType.BROADCAST,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=output_uuids,
                   opaques=None,
                   info=info)

    @classmethod
    def Free(cls, input_uuids, info=""):  # noqa
        return cls(opcode=PipelineInstType.FREE,
                   task_uuid=None,
                   input_uuids=input_uuids,
                   output_uuids=None,
                   opaques=None,
                   info=info,
                   print_uuids=False)

    def __str__(self):
        ret = ""
        ret += "Opcode: " + str(self.opcode)[17:] + ", Task uuid: " + str(
            self.task_uuid)
        if self.print_uuids:
            ret += ", input uuids:" + str(self.input_uuids)
            ret += ", output uuids:" + str(self.output_uuids)
        ret += ", Info: " + self.info
        return ret


AllocateZeroWorkerExecutableConfig = namedtuple(
    "AllocateZeroWorkerExecutableConfig",
    ["exec_uuid", "grad_shard_shapes", "grad_shard_dtypes"])
MemZeroWorkerExecutableConfig = namedtuple(
    "MemZeroWorkerExecutableConfig",
    ["exec_uuid", "grad_shard_shapes", "grad_shard_dtypes"])
PartialGradWorkerExecutableConfig = namedtuple(
    "PartialGradWorkerExecutableConfig",
    ["exec_uuid", "hlo_proto", "strategy_config", "grad_sync_channel_ids"])

ExecutableConfig = Union[AllocateZeroWorkerExecutableConfig,
                         MemZeroWorkerExecutableConfig,
                         PartialGradWorkerExecutableConfig]


def flatten_uuid_set(container):
    """Convert a nested array to an OrderedSet of elements in the array."""
    output = OrderedSet()
    for e in container:
        if isinstance(e, (np.ndarray, list)):
            output.update(flatten_uuid_set(e))
        else:
            output.add(e)
    return output


def _get_dict(d: Dict[Any, Dict], k) -> Dict:
    return d.setdefault(k, {})


class DecentralizedDistributedRuntime(BaseDistributedRuntime):
    """
    A decentralized pipeline_parallel runtime.

    This runtime uses the driver to compile and generate static instructions for each
    worker. It sends the instructions to distributed workers and launches the training.
    See the docstring of `BaseDistributedRuntime`.
    """

    def __init__(self,
                 *,
                 pipeline_stages: Sequence[XlaShardedPipelineComputation],
                 global_invars: Sequence[Var],
                 grad_dummy_invars: Sequence[Var],
                 global_outvars: Sequence[Var],
                 physical_meshes: DistributedPhysicalDeviceMeshGroup,
                 dependency: np.ndarray,
                 schedule: PipelineSchedule,
                 is_batch: Sequence[bool],
                 num_batch: int,
                 flop_count: int):
        super().__init__(pipeline_stages=pipeline_stages,
                         global_invars=global_invars,
                         grad_dummy_invars=grad_dummy_invars,
                         global_outvars=global_outvars,
                         physical_meshes=physical_meshes,
                         dependency=dependency,
                         schedule=schedule,
                         is_batch=is_batch,
                         num_batch=num_batch)
        self.uuid_counter = 0  # counter for local buffer uuid
        self.flop_count = flop_count

        # List[stage_idx -> executable_uuid]
        self.executable_uuids = [] 

        # Cached sharding indices for inputs.
        # List[mesh_idx -> List[sharding_indices]].
        self.input_indices = [] 
        # Whether the var should be donated
        # List[mesh_idx -> List[bool]]
        self.donate_invars = []
        # Whether the var should be deleted after shard
        # List[mesh_idx -> List[bool]]
        self.delete_after_shard = []
        # List[mesh_idx -> List[arg_idx]]
        self.mesh_arg_indices = []
        # List[arg_idx -> List[(mesh_idx, sharding_spec)]]
        self.batch_arg_on_mesh = []

        # Dict[worker -> List[uuid]]
        self.output_local_uuid_list = {}
        # List[arg_idx -> List[mesh_idx -> int]]
        self.mesh_output_indices = []
        # List[mesh_idx -> List[arg_idx -> sharding_spec]]
        self.output_spec_list = [[] for _ in range(self.num_mesh)]

        # List[stage_idx -> str]
        self.hlo_texts_after_spmd_partitioner = []

        # Compile pipeline instructions and configs of mesh executables
        (instruction_lists, executable_config_lists,
         input_local_uuid_lists, grad_uuids,
         accumulated_uuid_lists) = self._compile()

        # Create a PipelineMeshWorkerExecutable for each MeshHostWorker
        self.worker_executable_uuid_mapping = {}
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            mesh_grad_uuids = grad_uuids[mesh_idx]
            for worker_idx, worker in enumerate(physical_mesh.workers):
                acc_grad_local_uuids = []
                if len(mesh_grad_uuids) > 0:
                    acc_grad_local_uuids = mesh_grad_uuids[worker_idx]
                args = (instruction_lists[worker],
                        input_local_uuid_lists[worker],
                        self.output_local_uuid_list[worker],
                        executable_config_lists[worker],
                        acc_grad_local_uuids,
                        accumulated_uuid_lists[worker],
                        self.donate_invars[mesh_idx])
                uuid = next_mesh_executable_uuid()
                worker.put_executable.remote(uuid, PipelineMeshWorkerExecutable,
                                             *args)
                self.worker_executable_uuid_mapping[worker] = uuid

        # For handling input/outputs
        self.outs_handler: Callable = None
        self._setup_outs_handler()

    def get_next_uuids(self, num) -> np.ndarray:
        """Get the next uuids as a numpy array of uuids."""
        ret = np.arange(start=self.uuid_counter,
                        stop=self.uuid_counter + num,
                        dtype=np.int64)
        self.uuid_counter += num
        return ret

    def _compile(self):
        num_mesh = self.num_mesh

        instruction_lists = {}  # Dict[worker -> List[PipelineInstruction]]
        executable_config_lists = {}  # Dict[worker -> List[ExecutableConfig]]

        # Initialize instruction lists for workers.
        # Each worker has its own instruction list because resharding is not SPMD.
        for physical_mesh in self.physical_meshes:
            for worker in physical_mesh.workers:
                instruction_lists[worker] = []
                executable_config_lists[worker] = []

        # Dict[var_key -> Dict[mesh_idx -> np.array of uuids]]
        # The shape of the numpy array is [num_hosts, num_devices_per_host]
        var_at = {}

        # Compile forward, backward and apply_grad computations
        executable_uuids = self._compile_computation_executables(
            executable_config_lists)
        self.executable_uuids = executable_uuids 

        # Compile gradient buffer allocations
        grad_uuids = self._compile_grad_buffer_allocations(
            instruction_lists, executable_config_lists, var_at)

        # Split input into micro batches
        not_batch_invars = OrderedSet([
            var for var, batch in zip(self.global_invars, self.is_batch)
            if not batch
        ])
        input_local_uuid_lists = self._compile_split_input_to_microbatches(
            not_batch_invars, var_at)

        # Simulate the pipeline schedule and generate instructions
        worker_tmp_instructions = {}
        for mesh in self.physical_meshes:
            for worker in mesh.workers:
                worker_tmp_instructions[worker] = []

        donation_mapping = [DisjointDict() for _ in range(num_mesh)]
        worker_to_idx = {}
        for mesh_idx, mesh in enumerate(self.physical_meshes):
            for worker_idx, worker in enumerate(mesh.workers):
                worker_to_idx[worker] = (mesh_idx, worker_idx)

        def get_invar_key(invar, batch_idx):
            if invar in not_batch_invars:
                var_key = repr(invar)
                key = (repr(invar), 0)
            elif (invar in self.grad_dummy_invars and
                  batch_idx != self.schedule.first_backward_batch_index):
                var_key = self.grad_dummy_invars[invar]
                key = (var_key,
                       self.schedule.previous_backward_batch_index(batch_idx))
            else:
                var_key = repr(invar)
                key = (repr(invar), batch_idx)
            return var_key, key

        for _, sched in enumerate(self.schedule.schedules):
            for tmp_list in worker_tmp_instructions.values():
                tmp_list.clear()

            for mesh_idx, task in enumerate(sched):
                if not task:
                    continue
                physical_mesh = self.physical_meshes[mesh_idx]
                num_devices_per_host = physical_mesh.num_devices_per_host
                batch_idx, stage_idx = task
                stage = self.stages[stage_idx]
                received_keys = OrderedSet()
                # shard_args for intermediates
                to_reshard_vars = []
                reshard_sharding_specs = []
                for invar, spec in zip(stage.invars,
                                       stage.input_sharding_specs):
                    var_key, key = get_invar_key(invar, batch_idx)
                    if mesh_idx in var_at[key]:
                        # have a copy at the current mesh
                        continue
                    if len(var_at[key]) > 1:
                        raise NotImplementedError(
                            "Not support resharding replicated")
                    to_reshard_vars.append(invar)
                    reshard_sharding_specs.append(spec)
                keys = [
                    get_invar_key(var, batch_idx)[1] for var in to_reshard_vars
                ]
                if len(keys):
                    # TODO(yonghao): only compile alloc once, use multiple times
                    output_uuids = self._compile_alloc(to_reshard_vars,
                                                       reshard_sharding_specs,
                                                       mesh_idx, keys, False,
                                                       instruction_lists,
                                                       executable_config_lists,
                                                       var_at)
                    # shape: (args, num_hosts, num_devices_per_host)
                    transposed = output_uuids.transpose([1, 0, 2])
                    recv_uuid_list = transposed

                    for invar, recv_uuids in zip(to_reshard_vars,
                                                 recv_uuid_list):
                        var_key, key = get_invar_key(invar, batch_idx)
                        src_idx, src_uuids = list(var_at[key].items())[0]
                        resharding_task = self._resharding_tasks[src_idx][
                            mesh_idx][var_key]
                        if global_config.resharding_mode == "send_recv":
                            self._compile_resharding_task(
                                self.physical_meshes[src_idx],
                                self.physical_meshes[mesh_idx], src_uuids,
                                resharding_task, recv_uuids, instruction_lists)
                        else:
                            self._compile_broadcast_resharding_task(
                                self.physical_meshes[src_idx],
                                self.physical_meshes[mesh_idx], src_uuids,
                                resharding_task, recv_uuids, instruction_lists)
                        received_keys.add(key)

                # execute
                # allocate uuids for buffers created by RUN
                for outvar in stage.outvars:
                    key = (repr(outvar), batch_idx)
                    # get uuids of this outvar
                    _get_dict(var_at, key)[mesh_idx] = self.get_next_uuids(
                        physical_mesh.num_devices).reshape(
                            -1, num_devices_per_host)

                exec_uuid = executable_uuids[stage_idx]
                donated_invars = self.stages[stage_idx].donated_invars
                for worker_idx, worker in enumerate(physical_mesh.workers):
                    # Get input and output uuids. They should be at the mesh
                    input_uuids = np.zeros(
                        (len(stage.invars), num_devices_per_host),
                        dtype=np.int64)
                    output_uuids = np.zeros(
                        (len(stage.outvars), num_devices_per_host),
                        dtype=np.int64)
                    for idx, invar in enumerate(stage.invars):
                        _, key = get_invar_key(invar, batch_idx)
                        input_uuids[idx] = var_at[key][mesh_idx][worker_idx]
                    for idx, outvar in enumerate(stage.outvars):
                        key = (repr(outvar), batch_idx)
                        output_uuids[idx] = var_at[key][mesh_idx][worker_idx]
                    for idx in range(len(stage.invars)):
                        if donated_invars[idx]:
                            donation_mapping[mesh_idx].update(
                                input_uuids[idx], output_uuids[idx])

                    kwargs = {
                        "skip_grad_sync":
                            self.schedule.should_skip_grad_sync(task),
                        "sync_before": False,
                        "sync_after": False,
                    }

                    worker_tmp_instructions[worker].append(
                        PipelineInstruction.Run(exec_uuid,
                                                input_uuids,
                                                output_uuids,
                                                kwargs,
                                                info=f"stage {stage_idx}"))
                # free all received buffers
                # received_uuids = [
                #     var_at[key].pop(mesh_idx) for key in received_keys
                # ]
                # for worker_idx, worker in enumerate(physical_mesh.workers):
                #     instructions = worker_tmp_instructions[worker]
                #     for uuids in received_uuids:
                #         instructions.append(
                #             PipelineInstruction.Free(uuids[worker_idx]))
            for worker, worker_instruction in worker_tmp_instructions.items():
                instruction_lists[worker].extend(worker_instruction)

        # Compile information for outputs
        self._compile_collect_outputs(var_at)

        # Insert buffer free instructions
        accumulated_uuid_lists = {}
        for worker in instruction_lists:
            used_outside = flatten_uuid_set(self.output_local_uuid_list[worker])
            mesh_idx, worker_idx = worker_to_idx[worker]
            accumulated_uuids = grad_uuids[mesh_idx]
            if len(accumulated_uuids) > 0:
                accumulated_uuids = accumulated_uuids[worker_idx]
            accumulated_uuids = [[
                donation_mapping[mesh_idx].recursive_lookup(uuid)
                for uuid in uuids
            ] for uuids in accumulated_uuids]
            donated = set(donation_mapping[mesh_idx].keys())
            used_outside.update(flatten_uuid_set(accumulated_uuids))
            accumulated_uuid_lists[worker] = accumulated_uuids
            # pylint: disable=modified-iterating-dict
            instruction_lists[worker] = self._compile_free(
                worker, used_outside, donated, instruction_lists)

        return (instruction_lists, executable_config_lists,
                input_local_uuid_lists, grad_uuids,
                accumulated_uuid_lists)

    def _compile_computation_executables(self, executable_config_lists):
        """Compile executables for forward, backward, and apply_grad compuations."""
        executable_uuids = []  # List[stage_idx -> executable_uuids]

        for stage_idx, stage in enumerate(self.stages):
            exec_uuid = next_mesh_executable_uuid()
            executable_uuids.append(exec_uuid)

            mesh_idx = self.schedule.stage_placement(stage_idx)
            assert len(mesh_idx) == 1
            mesh_idx = list(mesh_idx)[0]
            hlo_module = stage.get_spmd_partitioned()
            hlo_proto = hlo_module.as_serialized_hlo_module_proto()
            for worker in self.physical_meshes[mesh_idx].workers:
                executable_config_lists[worker].append(
                    PartialGradWorkerExecutableConfig(
                        exec_uuid, hlo_proto, stage.strategy_config,
                        stage.output_acc_grad_indices))

        return executable_uuids

    def _compile_grad_buffer_allocations(self, instruction_lists,
                                         executable_config_lists, var_at):
        """Compile gradient buffer allocations."""
        num_mesh = len(self.physical_meshes)
        mesh_grad_vars = [{} for _ in range(num_mesh)]
        # TODO(yonghao): replicated code. abstract this part?
        # collect gradient accumulation buffers in each mesh
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            mesh_idx = mesh_indices[0]
            grad_var_spec_dict = mesh_grad_vars[mesh_idx]
            input_specs = stage.input_sharding_specs
            for var_idx, invar in enumerate(stage.invars):
                if invar in self.grad_dummy_invars:
                    if invar in grad_var_spec_dict:
                        raise NotImplementedError(
                            f"accumulate {invar} at multiple stages in a mesh")
                    grad_var_spec_dict[invar] = input_specs[var_idx]

        grad_uuids = [[] for _ in range(num_mesh)]
        for mesh_idx in range(num_mesh):
            grad_var_spec_dict = mesh_grad_vars[mesh_idx]
            if len(grad_var_spec_dict):
                grad_vars, grad_sharding_specs = list(
                    zip(*grad_var_spec_dict.items()))

                # TODO(yonghao): below, we start accumulation according to hints provided by schedule;
                #    this is a case that only works for pipeline-parallel training and gradients.
                #    In some model, some var has non-gradient intermediate states that need accumulation.
                #    for these vars, we need to record its first mb index when accum will take place.
                keys = [(repr(var), self.schedule.first_backward_batch_index)
                        for var in grad_vars]
                grad_uuids[mesh_idx] = self._compile_alloc(
                    grad_vars, grad_sharding_specs, mesh_idx,
                    keys, global_config.use_memzero_for_gradient_accumulation,
                    instruction_lists, executable_config_lists, var_at)

        return grad_uuids

    def _compile_split_input_to_microbatches(self, not_batch_invars, var_at):
        """
        Split batch arguments into micro batches.

        The split is like:
        before: a, b, c, d
        after (b, d are batch args and #mb=2): a, b0, b1, c, d0, d1
        """
        donated_invar_set = OrderedSet()
        global_invar_set = OrderedSet(self.global_invars)
        for stage in self.stages:
            for invar, donate in zip(stage.invars, stage.donated_invars):
                if donate and invar in global_invar_set:
                    donated_invar_set.add(invar)
        num_mesh = len(self.physical_meshes)
        global_indices = {}
        invar_counter = 0
        mesh_arg_lists = [None for _ in range(num_mesh)]
        self.batch_arg_on_mesh = [None] * len(self.global_invars)
        batch_arg_indices = {v: idx for idx, v in enumerate(self.global_invars)}

        # Split barch args
        for arg_idx, invar in enumerate(self.global_invars):
            if invar in not_batch_invars:
                key = invar, 0
                global_indices[key] = invar_counter
                invar_counter += 1
            else:
                self.batch_arg_on_mesh[arg_idx] = []
                for batch_idx in range(self.num_batch):
                    key = invar, batch_idx
                    global_indices[key] = invar_counter
                    invar_counter += 1

        # Dispatch args to each mesh
        arg_last_use = [-1] * invar_counter
        for mesh_idx in range(num_mesh):
            mesh_arg_set = OrderedSet()
            var_to_spec = {}
            mesh_batch_vars = OrderedSet()
            for stage_idx in self.schedule.mesh_stage_mapping[mesh_idx]:
                stage = self.stages[stage_idx]
                for spec, invar in zip(stage.input_sharding_specs,
                                       stage.invars):
                    if invar in self.global_invars:
                        var_to_spec[invar] = spec
                        if invar in not_batch_invars:
                            mesh_arg_set.add((invar, 0))
                        else:
                            for batch_idx in range(self.num_batch):
                                mesh_arg_set.add((invar, batch_idx))
                            mesh_batch_vars.add(invar)
            mesh_arg_list = list(mesh_arg_set)
            mesh_arg_lists[mesh_idx] = mesh_arg_list
            for invar in mesh_batch_vars:
                self.batch_arg_on_mesh[batch_arg_indices[invar]].append(
                    (mesh_idx, var_to_spec[invar]))

            self.donate_invars.append(
                [key[0] in donated_invar_set for key in mesh_arg_list])
            self.input_indices.append([
                pxla.spec_to_indices(key[0].aval.shape, var_to_spec[key[0]])
                for key in mesh_arg_list
            ])
            mesh_global_indices = [global_indices[key] for key in mesh_arg_list]
            for global_idx in mesh_global_indices:
                arg_last_use[global_idx] = mesh_idx
            self.mesh_arg_indices.append(mesh_global_indices)

        for mesh_idx in range(num_mesh):
            self.delete_after_shard.append([
                arg_last_use[idx] == mesh_idx and donate
                for idx, donate in zip(self.mesh_arg_indices[mesh_idx],
                                       self.donate_invars[mesh_idx])
            ])

        # Get local uuids for each input
        input_local_uuid_lists = defaultdict(list)
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            mesh_arg_list = mesh_arg_lists[mesh_idx]
            num_args = len(mesh_arg_list)
            # shape: (num_args, num_hosts, num_devices_per_host)
            arg_uuids = self.get_next_uuids(
                num_args * physical_mesh.num_devices).reshape(
                    num_args, -1, physical_mesh.num_devices_per_host)
            for arg_idx, key in enumerate(mesh_arg_lists[mesh_idx]):
                key = repr(key[0]), key[1]
                _get_dict(var_at, key)[mesh_idx] = arg_uuids[arg_idx]
                for worker_idx, worker in enumerate(physical_mesh.workers):
                    input_local_uuid_lists[worker].append(
                        arg_uuids[arg_idx, worker_idx])
        return input_local_uuid_lists

    def _compile_collect_outputs(self, var_at) -> None:
        """
        Generate output information.

        This function dispatches output information, including local uuid, local indices to global
        indices, and output specs to each mesh.
        """
        num_mesh = len(self.physical_meshes)

        for mesh in self.physical_meshes:
            for worker in mesh.workers:
                self.output_local_uuid_list[worker] = []
        # collect outvar specs
        var_to_spec_all_meshes = []
        global_outvar_set = OrderedSet(self.global_outvars)
        for mesh_idx in range(num_mesh):
            var_to_spec = {}
            for stage_idx in self.schedule.mesh_stage_mapping[mesh_idx]:
                stage = self.stages[stage_idx]
                for spec, outvar in zip(stage.output_sharding_specs,
                                        stage.outvars):
                    if outvar in global_outvar_set:
                        var_to_spec[outvar] = spec
            var_to_spec_all_meshes.append(var_to_spec)
        # assign indices and get specs
        for outvar in self.global_outvars:
            # the apply gradient only writes to microbatch 0
            key = (repr(outvar), self.schedule.last_backward_batch_index)
            var_meshes = var_at[key]
            mesh_out_indices = {}
            for mesh_idx in var_meshes:
                mesh = self.physical_meshes[mesh_idx]
                uuids = var_meshes[mesh_idx]
                for worker_idx, worker in enumerate(mesh.workers):
                    self.output_local_uuid_list[worker].append(
                        uuids[worker_idx])
                mesh_out_indices[mesh_idx] = (
                    len(self.output_local_uuid_list[worker]) - 1)
                self.output_spec_list[mesh_idx].append(
                    var_to_spec_all_meshes[mesh_idx][outvar])
            self.mesh_output_indices.append(mesh_out_indices)

    def _compile_alloc(self, variables, sharding_specs,
                       mesh_idx, keys, preallocated,
                       instruction_lists, executable_config_lists, var_at):
        """Compile an executable which allocates zero buffers.

        The zero buffers are:
        1) gradient accumulation buffers
        2) temp buffers for receiving tensors
        """
        config_class = (MemZeroWorkerExecutableConfig
                        if preallocated else AllocateZeroWorkerExecutableConfig)
        avals = [var.aval for var in variables]
        sharded_shapes = [
            get_shard_shape(aval, spec)
            for aval, spec in zip(avals, sharding_specs)
        ]
        dtypes = [aval.dtype for aval in avals]
        exec_uuid = next_mesh_executable_uuid()
        config = config_class(exec_uuid, sharded_shapes, dtypes)

        physical_mesh = self.physical_meshes[mesh_idx]
        output_uuids = self.get_next_uuids(
            len(variables) * physical_mesh.num_devices).reshape(
                len(physical_mesh.workers), len(variables), -1)
        for worker_idx, worker in enumerate(physical_mesh.workers):
            executable_config_lists[worker].append(config)
            if preallocated:
                in_uuids = output_uuids[worker_idx]
                out_uuids = []
            else:
                in_uuids = []
                out_uuids = output_uuids[worker_idx]
            instruction_lists[worker].append(
                PipelineInstruction.Run(config.exec_uuid,
                                        in_uuids,
                                        out_uuids, {
                                            "sync_before": False,
                                            "sync_after": False
                                        },
                                        info="mem set zero" if preallocated else
                                        "allocate zero for recv"))

        # shape: (#args, num_hosts, num_devices_per_host)
        transposed = output_uuids.transpose([1, 0, 2])
        for var_idx in range(len(variables)):
            key = keys[var_idx]
            _get_dict(var_at, key)[mesh_idx] = transposed[var_idx]
        return output_uuids

    # TODO(yonghao): set empty buffer is not compatiable with local allgather
    @staticmethod
    def _compile_resharding_task(src_mesh,
                                 dst_mesh,
                                 src_uuids: np.ndarray,
                                 resharding_task: SymbolicReshardingTask,
                                 recv_uuids: np.ndarray,
                                 instruction_lists,
                                 set_empty_buffer=False):
        """
        Compile and generate SEND and RECV PipelineInstructions for a ReshardingTask.

        Args:
            src_mesh: the src mesh
            dst_mesh: the dst mesh
            src_uuids: uuids of resharded buffer in src mesh
            resharding_task: the task to be compiled
            recv_uuids: uuids of resharded buffer in dst mesh
            set_empty_buffer: set the empty buffer when recv or not
        """
        num_devices_per_host = dst_mesh.num_devices_per_host
        send_buf_uuids = {worker: [] for worker in src_mesh.workers}
        recv_buf_uuids = {worker: [] for worker in dst_mesh.workers}
        num_device_sender_host = src_mesh.num_devices_per_host

        # collect uuids of each send_tile in each worker based on resharding_task's plan
        for sender_str in resharding_task.sender_uuid_plan:
            send_worker = resharding_task.collective_group.device_str_to_mesh_worker_map[
                sender_str]
            send_buf_flat_idx = resharding_task.task_spec.src.device_str_to_flat_index[
                sender_str]
            send_buf_host = send_buf_flat_idx // num_device_sender_host
            send_buf_device = send_buf_flat_idx % num_device_sender_host
            send_buf_uuids[send_worker].append(src_uuids[send_buf_host,
                                                         send_buf_device])

        # add send tasks for each worker
        for w, task_uuid in resharding_task.send_worker_task_ids.items():
            input_uuids = send_buf_uuids[w]
            instruction_lists[w].append(
                PipelineInstruction.Send(task_uuid, input_uuids))

        # collect uuids of each recv_tile in each worker based on resharding_task's plan
        for receiver_str in resharding_task.receiver_uuid_plan:
            receiver_worker = resharding_task.collective_group.device_str_to_mesh_worker_map[
                receiver_str]
            recv_buf_flat_idx = resharding_task.task_spec.dst.device_str_to_flat_index[
                receiver_str]
            recv_buf_host = recv_buf_flat_idx // num_devices_per_host
            recv_buf_device = recv_buf_flat_idx % num_devices_per_host
            recv_buf_uuids[receiver_worker].append(recv_uuids[recv_buf_host,
                                                              recv_buf_device])

        # add recv task for each worker
        for w, task_uuid in resharding_task.recv_worker_task_ids.items():
            output_uuids = recv_buf_uuids[w]
            allgather_uuid = (resharding_task.allgather_worker_task_ids[w] if
                              resharding_task.is_local_allgather_task else None)
            instruction_lists[w].append(
                PipelineInstruction.Recv(task_uuid, output_uuids,
                                         set_empty_buffer, allgather_uuid))

    @staticmethod
    def _compile_broadcast_resharding_task(src_mesh,
                                           dst_mesh,
                                           src_uuids: np.ndarray,
                                           resharding_task: SymbolicBroadcastReshardingTask,
                                           recv_uuids: np.ndarray,
                                           instruction_lists,
                                           set_empty_buffer=False):

        # add broadcast-based resharding task for each worker
        for w, task_uuid in resharding_task.broadcast_worker_task_ids.items():
            output_uuids = None
            input_uuids = None
            if w in src_mesh.workers:
                host_id = src_mesh.workers.index(w)
                input_uuids = src_uuids[host_id]
            else:
                host_id = dst_mesh.workers.index(w)
                output_uuids = recv_uuids[host_id]
            instruction_lists[w].append(PipelineInstruction.Broadcast(task_uuid, 
                                                                      input_uuids, 
                                                                      output_uuids, 
                                                                      "broadcast"))

    @staticmethod
    def _compile_free(worker, used_outside, donated, instruction_lists):
        """Compile and generate FREE PipelineInstruction to recycle memory."""
        instruction_list = instruction_lists[worker]
        new_list = []
        cannot_free_uuids = OrderedSet(used_outside)
        cannot_free_uuids.update(donated)
        for instruction in reversed(instruction_list):
            # for free instruction, do not free again
            if instruction.input_uuids is None:
                new_list.append(instruction)
                continue
            input_uuids = flatten_uuid_set(instruction.input_uuids)
            if not instruction.opcode == PipelineInstType.FREE:
                unused_uuids = input_uuids.difference(cannot_free_uuids)
                if len(unused_uuids) > 0:
                    new_list.append(
                        PipelineInstruction.Free(np.array(list(unused_uuids))))
            cannot_free_uuids.update(input_uuids)
            new_list.append(instruction)
        return list(reversed(new_list))

    def _exec_split_args(self, args, batch_dim=0):
        split_args = []
        num_batch = self.num_batch
        for arg_idx, arg in enumerate(args):
            if self.is_batch[arg_idx]:
                # dispatch and split on worker.
                replicas = [None] * num_batch
                for mesh_and_shard in self.batch_arg_on_mesh[arg_idx]:
                    mesh_idx, sharding_spec = mesh_and_shard
                    mesh = self.physical_meshes[mesh_idx]
                    splits = mesh.shard_batch_arg(
                        arg, sharding_spec, num_batch, batch_dim,
                        self.global_invars[arg_idx].aval)
                    for batch_idx, split in enumerate(splits):
                        if replicas[batch_idx] is not None:
                            replicas[batch_idx].add_replica(mesh, split)
                        else:
                            replicas[batch_idx] = ReplicatedDistributedArray(
                                [mesh], [split])
                split_args.extend(replicas)
            else:
                split_args.append(arg)
        return split_args

    def run(self, *args):
        """The run function that maps to train_step()."""
        input_bufs: List[Any] = [None for _ in range(self.num_mesh)]
        input_uuids: List[Any] = [None for _ in range(self.num_mesh)]
        output_bufs: List[Any] = [None for _ in range(self.num_mesh)]
        output_uuids: List[Any] = [None for _ in range(self.num_mesh)]

        num_outs = [
            len(self.output_local_uuid_list[mesh.workers[0]])
            for mesh in self.physical_meshes
        ]

        # Shard inputs
        split_args = self._exec_split_args(args)
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            mesh_args = [
                split_args[idx] for idx in self.mesh_arg_indices[mesh_idx]
            ]
            input_bufs[mesh_idx] = physical_mesh.shard_args_to_bufs(
                self.input_indices[mesh_idx], self.delete_after_shard[mesh_idx],
                mesh_args)
            num_hosts = physical_mesh.num_hosts
            num_devices_per_host = physical_mesh.num_devices_per_host
            input_uuids[mesh_idx] = (get_uuid_np_array(
                input_bufs[mesh_idx]).reshape(len(mesh_args), num_hosts,
                                              num_devices_per_host).transpose(
                                                  [1, 0, 2]))
            output_uuids[mesh_idx] = next_remote_buffer_uuid(
                num_hosts * num_outs[mesh_idx] * num_devices_per_host).reshape(
                    num_hosts, num_outs[mesh_idx], num_devices_per_host)
            #print(f"after shard_args mesh_idx={mesh_idx} allocated:",
            #       ray.get(physical_mesh.workers[0].get_memory_allocated.remote()) / 1024**3, "max_allocated:",
            #       ray.get(physical_mesh.workers[0].get_max_memory_allocated.remote()) / 1024**3)

        # Execute
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            for i, worker in enumerate(physical_mesh.workers):
                worker.run_executable.remote(
                    self.worker_executable_uuid_mapping[worker],
                    input_uuids[mesh_idx][i], output_uuids[mesh_idx][i],
                    sync_for_timer=True)

        # Handle donation
        for mesh_idx in range(len(self.physical_meshes)):
            inputs = input_bufs[mesh_idx]
            for bufs, donate in zip(inputs, self.donate_invars[mesh_idx]):
                if donate:
                    for buf in bufs:
                        buf.set_deleted_on_workers()

        # Construct output_bufs
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            num_devices_per_host = physical_mesh.num_devices_per_host
            output_uuid_transposed = output_uuids[mesh_idx].transpose([1, 0, 2])
            output_bufs[mesh_idx] = np.empty(
                (num_outs[mesh_idx], physical_mesh.num_devices), dtype=object)
            for i in range(num_outs[mesh_idx]):
                for j in range(physical_mesh.num_devices):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    # dtype = self.global_outvars[
                    #     self.mesh_index_to_outvar_indices_mapping[mesh_idx]
                    #     [i]].aval.dtype
                    output_bufs[mesh_idx][i][j] = RemoteBufferRef(
                        physical_mesh, host_id, device_id,
                        output_uuid_transposed[i][host_id][device_id])

        # Check if there is OOM
        if global_config.pipeline_check_alive:
            self._check_alive()

        return self.outs_handler(output_bufs)

    def _setup_outs_handler(self):
        """Setup outs handlers that assemble RemoteBufs into DistributedArrays."""
        avals = [outvar.aval for outvar in self.global_outvars]
        is_replicated = [
            bool(len(self.outvar_index_to_mesh_index_mapping[i]) > 1)
            for i, _ in enumerate(self.global_outvars)
        ]

        def outs_handler(bufs):
            ret = []
            for i, _ in enumerate(avals):
                aval = avals[i]
                if not is_replicated[i]:
                    # construct DistributedArray
                    mesh_idx = self.outvar_index_to_mesh_index_mapping[i][0]
                    device_mesh = self.physical_meshes[mesh_idx]
                    outvar_index_on_mesh = self.mesh_output_indices[i][mesh_idx]
                    spec = self.output_spec_list[mesh_idx][outvar_index_on_mesh]
                    arr = DistributedArray(
                        device_mesh=device_mesh,
                        aval=aval,
                        sharding_spec=spec,
                        remote_buffers=bufs[mesh_idx][outvar_index_on_mesh],
                        indices=pxla.spec_to_indices(aval.shape, spec))
                else:
                    # otherwise, construct RepliatedDistributedArray
                    meshes = []
                    distributed_arrays = []
                    for _, mesh_idx in enumerate(
                            self.outvar_index_to_mesh_index_mapping[i]):
                        meshes.append(self.physical_meshes[mesh_idx])
                        outvar_index_on_mesh = self.mesh_output_indices[i][
                            mesh_idx]
                        spec = self.output_spec_list[mesh_idx][
                            outvar_index_on_mesh]
                        distributed_arrays.append(
                            DistributedArray(
                                device_mesh=self.physical_meshes[mesh_idx],
                                aval=aval,
                                sharding_spec=spec,
                                remote_buffers=bufs[mesh_idx]
                                [outvar_index_on_mesh],
                                indices=pxla.spec_to_indices(aval.shape, spec)))
                    arr = ReplicatedDistributedArray(meshes, distributed_arrays)
                ret.append(arr)
            return ret

        self.outs_handler = outs_handler

    @cached_property
    def mesh_index_to_outvar_indices_mapping(self) -> Dict[int, List[int]]:
        """
        A mapping from mesh index to its related global invar index.

        Returns:
            mapping (Dict[int, List[int]]): mapping[mesh_idx] is a list containing
                the indices of global outvars of this mesh.
        """
        if self.mesh_output_indices is None:
            raise RuntimeError()
        mapping = {}
        for i, _ in enumerate(self.global_outvars):
            mesh_out_indices = self.mesh_output_indices[i]
            for mesh_idx in mesh_out_indices:
                if mesh_idx not in mapping:
                    mapping[mesh_idx] = []
                mapping[mesh_idx].append(i)
        return mapping

    @cached_property
    def outvar_index_to_mesh_index_mapping(self) -> Dict[int, List[int]]:
        """
        A mapping from an outvar to the indices of meshes it locates on.

        Returns:
            mapping (Dict[int, List[int]]): mapping[outvar_idx] is a list
                containing the indices of meshes it locates on.
        """
        if self.mesh_output_indices is None:
            raise RuntimeError()
        mapping = {}
        for i, _ in enumerate(self.global_outvars):
            mapping[i] = list(self.mesh_output_indices[i].keys())
        return mapping

    def _debug_check(self):
        for mesh in self.physical_meshes:
            num_outs = -1
            for worker in mesh.workers:
                if num_outs == -1:
                    num_outs = len(self.output_local_uuid_list[worker])
                else:
                    assert len(self.output_local_uuid_list[worker]) == num_outs

    def get_execution_time_costs(self,
                                 warmup=2,
                                 timer_name="overall",
                                 return_all_costs=False):
        """Get the execution time cost given a timer name."""
        if timer_name not in timer_names:
            raise RuntimeError(
                f"Unrecognized timer name for pipeline parallel runtime. "
                f"Query timer name from the following: {timer_names.keys()}.")
        mesh_costs = []
        for mesh in self.physical_meshes:
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
            for mesh in self.physical_meshes:
                mesh.reset_remote_timer(name)

    def get_total_allocation_size(self):
        """Get the total allocated memory size of each mesh."""
        # TODO: compute the theoretical total allocation size
        raise NotImplementedError()

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
                physical_mesh = self.physical_meshes[mesh_idx]
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

    def _check_alive(self):
        try:
            rets = [
                worker.check_alive.remote()
                for mesh in self.physical_meshes
                for worker in mesh.workers
            ]
            ray.get(rets)
        except ray.exceptions.RayActorError:
            self._exception_shutdown()

    def shutdown(self):
        """Shutdown the runtime and recycle resources."""
        self.reset_benchmark_timers()
        self.physical_meshes.shutdown()

    def _exception_shutdown(self):
        """In this shutdown, some actors might have died."""
        self.physical_meshes.exception_shutdown()

    def profile_all_executables(self):
        """Profile all executables in the runtime."""
        all_profiled_handles = []
        for _, physical_mesh in enumerate(self.physical_meshes):
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


class PipelineMeshWorkerExecutable:
    """An executable that executes static pipeline runtime instructions on a worker."""

    def __init__(self, worker: MeshHostWorker, uuid: int,
                 instructions: Sequence[PipelineInstruction],
                 input_local_uuids: Sequence[int],
                 output_local_uuids: Sequence[int],
                 executable_configs: Sequence[ExecutableConfig],
                 acc_local_uuids: np.ndarray,
                 acc_out_uuids: Sequence[Sequence[int]],
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
        self.acc_in_uuids = [list(uuids) for uuids in list(acc_local_uuids)]
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
            # print(f"memory_allocated: {self.worker.get_memory_allocated()/1024**3:.3f} GB  "
            #       f"max_memory_allocated: {self.worker.get_max_memory_allocated()/1024**3:.3f} GB "
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
                self.worker.run_resharding_broadcast_task(instruction.task_uuid,
                                                          instruction.input_uuids
                                                          if instruction.input_uuids is not None
                                                          else instruction.output_uuids)
                timers("resharding_broadcast").suspend()
            elif instruction.opcode == PipelineInstType.FREE:
                timers("free").start()
                self.worker.delete_buffers(instruction.input_uuids)
                timers("free").suspend()

        for timer_name in [
                "compute", "resharding_send", "resharding_recv", "resharding_broadcast", "free"
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
            (np.mean(self.worker.profile_executable_with_dummy_inputs(exec_id)),
             self.worker.get_exec_total_allocation_size(exec_id) / 1024**3)
            for exec_id in self.partial_grad_exec_uuids
        }
        self.worker.reset_memory_stats()
        return ret

    def __del__(self):
        for exec_id in self._related_exec_uuids:
            self.worker.delete_executable(exec_id)
