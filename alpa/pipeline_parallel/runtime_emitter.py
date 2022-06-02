"""Compile stages to pipeline instructions."""
from collections import namedtuple, defaultdict
from dataclasses import dataclass
import enum
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from jax._src.tree_util import PyTreeDef, tree_unflatten
from jax.core import Var
from jax.interpreters import pxla
from jax.lib import xla_bridge as xb
import numpy as np

from alpa.global_env import global_config
from alpa.device_mesh import DistributedArray, PhysicalDeviceMeshGroup, ReplicatedDistributedArray
from alpa.mesh_executable import next_mesh_executable_uuid
from alpa.pipeline_parallel.computation import XlaShardedPipelineComputation
from alpa.pipeline_parallel.schedules import PipelineSchedule
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CrossMeshCommunicator, SymbolicBroadcastReshardingTask,
    SymbolicReshardingTask)
from alpa.serialization import LoadInfo
from alpa.util import (DisjointDict, OrderedSet, cached_property,
                       get_shard_shape, get_microbatch_sharding_spec,
                       compile_concatenate)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    def Recv(cls,  # noqa
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
    def Broadcast(cls,  # noqa
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
ConcatWorkerExecutableConfig = namedtuple("ConcatWorkerExecutableConfig",
                                          ["exec_uuid", "hlo_proto"])
PartialGradWorkerExecutableConfig = namedtuple(
    "PartialGradWorkerExecutableConfig",
    ["exec_uuid", "hlo_proto", "strategy_config", "grad_sync_channel_ids"])

ExecutableConfig = Union[AllocateZeroWorkerExecutableConfig,
                         MemZeroWorkerExecutableConfig,
                         PartialGradWorkerExecutableConfig,
                         ConcatWorkerExecutableConfig]


def flatten_uuid_set(container):
    """Convert a nested array to an OrderedSet of elements in the array."""
    output = OrderedSet()
    for e in container:
        if isinstance(e, (np.ndarray, list)):
            output.update(flatten_uuid_set(e))
        else:
            output.add(e)
    return output


class PipelineInstEmitterHelper:

    def __init__(self, global_invars, grad_dummy_invars, is_batch,
                 schedule: PipelineSchedule):
        self.global_invars = global_invars
        self.global_batch_invars = OrderedSet(
            v for v, b in zip(global_invars, is_batch) if b)
        self.grad_dummy_invars = grad_dummy_invars
        self.schedule = schedule
        # Dict[var_key -> Dict[mesh_idx -> np.array of uuids]]
        # The shape of the numpy array is [num_hosts, num_devices_per_host]
        self.env = {}

    def _get_var_key(self, var, batch_idx):
        if var in self.global_invars and var not in self.global_batch_invars:
            key = (var, 0)
        elif (var in self.grad_dummy_invars and
              batch_idx != self.schedule.first_backward_batch_index):
            key = (self.grad_dummy_invars[var],
                   self.schedule.previous_backward_batch_index(batch_idx))
        else:
            key = (var, batch_idx)
        return key

    def get_var_with_accumulate(self, var, batch_idx):
        if (var in self.grad_dummy_invars and
                batch_idx != self.schedule.first_backward_batch_index):
            return self.grad_dummy_invars[var]
        else:
            return var

    def get_var_mesh_uuids(self, var, batch_idx, mesh_idx) -> np.ndarray:
        key = self._get_var_key(var, batch_idx)
        return self.env[key][mesh_idx]

    def get_var_meshes(self, var, batch_idx) -> Dict[int, np.ndarray]:
        key = self._get_var_key(var, batch_idx)
        return self.env.setdefault(key, {})

    def set_var_mesh_uuids(self, var, batch_idx, mesh_idx, uuids) -> None:
        key = self._get_var_key(var, batch_idx)
        self.env.setdefault(key, {})[mesh_idx] = uuids

    def var_at(self, var, batch_idx, mesh_idx) -> bool:
        key = self._get_var_key(var, batch_idx)
        return mesh_idx in self.env.setdefault(key, {})


@dataclass
class PipeshardInputConfig:
    input_local_uuid_lists: Dict[Any, Sequence[np.ndarray]]
    donate_invars: Sequence[Sequence[bool]]
    mesh_arg_indices: Sequence[Sequence[int]]
    input_shard_indices: Sequence[Sequence[Any]]
    delete_after_shard: Sequence[Sequence[bool]]
    batch_invars: Sequence[Sequence[bool]]


# TODO(yonghao): use worker_idx as the dict's key
@dataclass
class PipeshardConfig:
    instruction_lists: Dict[Any, Sequence[PipelineInstruction]]
    # Executable configs
    executable_configs: Dict[Any, Sequence[ExecutableConfig]]
    executable_uuids: Sequence[int]
    # Input configs
    input_config: PipeshardInputConfig
    grad_uuids: Sequence[np.ndarray]
    reduced_var_uuid_lists: Dict[Any, np.ndarray]
    device_str_groups: Sequence[Sequence[OrderedSet]]
    # Output configs
    output_local_uuid_list: Sequence[Sequence]
    # Others
    outs_handler: Callable
    load_info: LoadInfo
    resharding_task_iter: Iterator

    @property
    def input_local_uuid_lists(self):
        return self.input_config.input_local_uuid_lists

    @property
    def donate_invars(self):
        return self.input_config.donate_invars

    @property
    def mesh_arg_indices(self):
        return self.input_config.mesh_arg_indices

    @property
    def input_shard_indices(self):
        return self.input_config.input_shard_indices

    @property
    def delete_after_shard(self):
        return self.input_config.delete_after_shard

    @property
    def batch_invars(self):
        return self.input_config.batch_invars


class PipelineInstEmitter:

    def __init__(self, *, stages: Sequence[XlaShardedPipelineComputation],
                 global_invars: Sequence[Var], grad_dummy_invars: Sequence[Var],
                 concat_vars_mapping: Dict[Var, Var],
                 global_outvars: Sequence[Var],
                 mesh_group: PhysicalDeviceMeshGroup,
                 schedule: PipelineSchedule, is_batch: Sequence[bool],
                 num_batch: int, in_tree: PyTreeDef):
        ##### Input arguments #####
        self.stages = stages
        self.global_invars = global_invars
        self.grad_dummy_invars = grad_dummy_invars
        self.concat_vars_mapping = concat_vars_mapping
        self.global_outvars = global_outvars
        self.mesh_group = mesh_group
        self.schedule = schedule
        self.is_batch = is_batch
        self.num_batch = num_batch
        self.in_tree = in_tree

        ##### Internal states #####
        self.uuid_counter = 0  # counter for local buffer uuid
        self.env = PipelineInstEmitterHelper(global_invars, grad_dummy_invars,
                                             is_batch, schedule)

        # for cross mesh communications
        self.num_mesh = len(mesh_group)

        ##### For handling outputs of the executable ####
        # List[arg_idx -> List[mesh_idx -> int]]
        self.mesh_output_indices = []
        # List[mesh_idx -> List[arg_idx -> sharding_spec]]
        self.output_spec_list = [[] for _ in range(len(mesh_group))]

        ##### For cross mesh communication ####
        self._precompile_sharding_specs()
        self._communicator = CrossMeshCommunicator(self.stages, self.schedule)
        self.num_mesh = len(mesh_group)
        self._resharding_tasks = [
            [{} for _ in range(self.num_mesh)] for _ in range(self.num_mesh)
        ]

    def _get_next_uuids(self, num) -> np.ndarray:
        """Get the next uuids as a numpy array of uuids."""
        ret = np.arange(start=self.uuid_counter,
                        stop=self.uuid_counter + num,
                        dtype=np.int64)
        self.uuid_counter += num
        return ret

    def _precompile_sharding_specs(self):
        """Run spmd partitioner pass for each stage to get sharding specs."""
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            stage.get_spmd_partitioned()

    def _compile_resharding_tasks(self):
        """Create and compile all resharding (send/recv/allgather) tasks."""
        for (src_mesh_idx, dst_mesh_idx,
             var_spec_map) in self._communicator.task_spec_iter():
            for var, spec in var_spec_map.items():
                cg = self.mesh_group.collective_groups[src_mesh_idx][
                    dst_mesh_idx]
                src_mesh = self.mesh_group[src_mesh_idx]
                dst_mesh = self.mesh_group[dst_mesh_idx]
                # TODO(yonghao): delay put_resharding_XXXX_task until pipeshard executable
                if global_config.resharding_mode == "send_recv":
                    self._resharding_tasks[src_mesh_idx][dst_mesh_idx][
                        var] = SymbolicReshardingTask(spec, cg, src_mesh,
                                                      dst_mesh)
                else:
                    self._resharding_tasks[src_mesh_idx][dst_mesh_idx][
                        var] = SymbolicBroadcastReshardingTask(
                            spec, cg, src_mesh, dst_mesh)

    def _establish_nccl_groups(self):
        """
        Identify NCCL groups based on resharding specs but do not instantiate them.

        We establish one collective group between two physical meshes, covering all the devices in
        these two meshes that require NCCL communication.

        Returns:
            device_str_groups (List[List[set]]): a num_mesh x num_mesh matrix. Only entries at
                device_str_groups[i][j] (i < j) are filled, entries with i > j are None, because
                (spec[i][j], spec[j][i]) will share collective groups.
        """
        device_str_groups = [[OrderedSet()
                              for _ in range(self.num_mesh)]
                             for _ in range(self.num_mesh)]
        # Merge (i, j) and (j, i)
        for i, j, var_spec_map in self._communicator.task_spec_iter():
            participants = OrderedSet()
            for _, spec in var_spec_map.items():  # for each var
                participants = participants | spec.get_participant_device_strs()
            if i <= j:
                device_str_groups[i][j] = device_str_groups[i][j] | participants
            else:
                device_str_groups[j][i] = device_str_groups[j][i] | participants

        # construct groups
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if i >= j:
                    assert not device_str_groups[i][j]
                    continue
                if not device_str_groups[i][j]:
                    continue
                self.mesh_group.establish_nccl_group(i, j, instantiate=False)
        return device_str_groups

    def compile(self):
        """Compile pipeline instructions and executables for workers."""
        num_mesh = len(self.mesh_group)
        device_str_groups = self._establish_nccl_groups()

        self._compile_resharding_tasks()
        # Compile forward, backward and apply_grad computations
        (executable_uuids,
         executable_config_lists) = self._compile_computation_executables()

        # Compile gradient buffer allocations
        grad_uuids, instruction_lists = self._compile_grad_buffer_allocations(
            executable_config_lists)

        # Split input into micro batches
        global_batch_invar_set = OrderedSet([
            var for var, batch in zip(self.global_invars, self.is_batch)
            if batch
        ])
        (input_config, input_shard_specs
         ) = self._compile_split_input_to_microbatches(global_batch_invar_set)

        # Simulate the pipeline schedule and generate instructions
        donation_mapping = [DisjointDict() for _ in range(num_mesh)]
        worker_to_idx = {}
        for mesh_idx, mesh in enumerate(self.mesh_group):
            for worker_idx, worker in enumerate(mesh.workers):
                worker_to_idx[worker] = (mesh_idx, worker_idx)

        for _, sched in enumerate(self.schedule.schedules):
            self._compile_exec_one_tick(sched, donation_mapping,
                                        instruction_lists, executable_uuids,
                                        executable_config_lists)

        # Compile concate
        self._compile_concate(instruction_lists, executable_config_lists)

        # Compile information for outputs
        output_local_uuid_list = self._compile_collect_outputs()

        # Insert buffer free instructions
        reduced_var_uuid_lists = {}
        for worker in instruction_lists:
            used_outside = flatten_uuid_set(output_local_uuid_list[worker])
            mesh_idx, worker_idx = worker_to_idx[worker]
            reduced_var_uuids = grad_uuids[mesh_idx]
            if len(reduced_var_uuids) > 0:
                reduced_var_uuids = reduced_var_uuids[worker_idx]
            reduced_var_uuids = np.array([[
                donation_mapping[mesh_idx].recursive_lookup(uuid)
                for uuid in uuids
            ] for uuids in reduced_var_uuids])
            donated = set(donation_mapping[mesh_idx].keys())
            used_outside.update(flatten_uuid_set(reduced_var_uuids))
            reduced_var_uuid_lists[worker] = reduced_var_uuids
            instruction_lists[worker] = self._compile_free(
                worker, used_outside, donated, instruction_lists)

        outs_handler = self._get_outs_handler()

        load_info = self._compile_load_info(input_config.mesh_arg_indices,
                                            input_shard_specs)
        return PipeshardConfig(
            instruction_lists=instruction_lists,
            # Executable configs
            executable_configs=executable_config_lists,
            executable_uuids=executable_uuids,
            # Input configs
            input_config=input_config,
            grad_uuids=grad_uuids,
            reduced_var_uuid_lists=reduced_var_uuid_lists,
            device_str_groups=device_str_groups,
            # Output configs
            output_local_uuid_list=output_local_uuid_list,
            # Others
            outs_handler=outs_handler,
            load_info=load_info,
            resharding_task_iter=self.resharding_task_iter())

    def resharding_task_iter(self):
        """An iterator over all resharding tasks."""
        for src_idx in range(self.num_mesh):
            for dst_idx in range(self.num_mesh):
                tasks = self._resharding_tasks[src_idx][dst_idx].values()
                for task in tasks:
                    yield task

    def _compile_get_vars_from_mesh(self, invars, dst_specs, mesh_idx,
                                    batch_idx, instruction_lists,
                                    executable_config_lists):
        if len(invars) == 0:
            return
        # TODO(yonghao): only compile alloc once, use multiple times
        output_uuids = self._compile_alloc(invars, dst_specs, mesh_idx,
                                           batch_idx, False, instruction_lists,
                                           executable_config_lists)
        # shape: (args, num_hosts, num_devices_per_host)
        transposed = output_uuids.transpose([1, 0, 2])
        recv_uuid_list = transposed

        for invar, recv_uuids in zip(invars, recv_uuid_list):
            var_key = self.env.get_var_with_accumulate(invar, batch_idx)
            src_idx, src_uuids = list(
                self.env.get_var_meshes(invar, batch_idx).items())[0]
            resharding_task = self._resharding_tasks[src_idx][mesh_idx][var_key]
            if global_config.resharding_mode == "send_recv":
                self._compile_resharding_task(self.mesh_group[src_idx],
                                              self.mesh_group[mesh_idx],
                                              src_uuids, resharding_task,
                                              recv_uuids, instruction_lists)
            else:
                self._compile_broadcast_resharding_task(
                    self.mesh_group[src_idx], self.mesh_group[mesh_idx],
                    src_uuids, resharding_task, recv_uuids, instruction_lists)

    def _compile_exec_one_tick(self, sched, donation_mapping, instruction_lists,
                               executable_uuids, executable_config_lists):
        worker_tmp_instructions = {}
        for mesh in self.mesh_group:
            for worker in mesh.workers:
                worker_tmp_instructions[worker] = []

        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            physical_mesh = self.mesh_group[mesh_idx]
            num_devices_per_host = physical_mesh.num_devices_per_host
            batch_idx, stage_idx = task
            stage = self.stages[stage_idx]
            # shard_args for intermediates
            to_reshard_vars = []
            reshard_sharding_specs = []
            for invar, spec in zip(stage.invars, stage.input_sharding_specs):
                if self.env.var_at(invar, batch_idx, mesh_idx):
                    # have a copy at the current mesh
                    continue
                if len(self.env.get_var_meshes(invar, batch_idx)) > 1:
                    raise NotImplementedError(
                        "Not support resharding replicated")
                to_reshard_vars.append(invar)
                reshard_sharding_specs.append(spec)
            self._compile_get_vars_from_mesh(to_reshard_vars,
                                             reshard_sharding_specs, mesh_idx,
                                             batch_idx, instruction_lists,
                                             executable_config_lists)

            # execute
            # allocate uuids for buffers created by RUN
            for outvar in stage.outvars:
                # get uuids of this outvar

                output_uuids = self._get_next_uuids(
                    physical_mesh.num_devices).reshape(-1, num_devices_per_host)
                self.env.set_var_mesh_uuids(outvar, batch_idx, mesh_idx,
                                            output_uuids)

            exec_uuid = executable_uuids[stage_idx]
            donated_invars = self.stages[stage_idx].donated_invars

            input_uuids = np.zeros((len(stage.invars), *physical_mesh.shape),
                                   dtype=np.int64)
            output_uuids = np.zeros((len(stage.outvars), *physical_mesh.shape),
                                    dtype=np.int64)
            for idx, invar in enumerate(stage.invars):
                input_uuids[idx] = self.env.get_var_mesh_uuids(
                    invar, batch_idx, mesh_idx)
            for idx, outvar in enumerate(stage.outvars):
                output_uuids[idx] = self.env.get_var_mesh_uuids(
                    outvar, batch_idx, mesh_idx)
            input_uuids = input_uuids.transpose([1, 0, 2])
            output_uuids = output_uuids.transpose([1, 0, 2])

            for worker_idx, worker in enumerate(physical_mesh.workers):
                # Get input and output uuids. They should be at the mesh
                worker_input_uuids = input_uuids[worker_idx]
                worker_output_uuids = output_uuids[worker_idx]
                for idx in range(len(stage.invars)):
                    if donated_invars[idx]:
                        donation_mapping[mesh_idx].update(
                            worker_input_uuids[idx], worker_output_uuids[idx])

                kwargs = {
                    "skip_grad_sync": self.schedule.should_skip_grad_sync(task),
                    "sync_before": False,
                    "sync_after": False,
                }

                worker_tmp_instructions[worker].append(
                    PipelineInstruction.Run(exec_uuid,
                                            worker_input_uuids,
                                            worker_output_uuids,
                                            kwargs,
                                            info=f"stage {stage_idx}"))

        for worker, worker_instruction in worker_tmp_instructions.items():
            instruction_lists[worker].extend(worker_instruction)

    def _compile_computation_executables(self):
        """Compile executables for forward, backward, and apply_grad compuations."""
        executable_uuids = []  # List[stage_idx -> executable_uuids]
        executable_config_lists = defaultdict(
            list)  # Dict[worker -> List[ExecutableConfig]]

        for stage_idx, stage in enumerate(self.stages):
            exec_uuid = next_mesh_executable_uuid()
            executable_uuids.append(exec_uuid)

            mesh_idx = self.schedule.stage_placement(stage_idx)
            assert len(mesh_idx) == 1
            mesh_idx = list(mesh_idx)[0]
            hlo_module = stage.get_spmd_partitioned()
            hlo_proto = hlo_module.as_serialized_hlo_module_proto()
            exec_config = PartialGradWorkerExecutableConfig(
                exec_uuid, hlo_proto, stage.strategy_config,
                stage.output_acc_grad_indices)
            for worker in self.mesh_group[mesh_idx].workers:
                executable_config_lists[worker].append(exec_config)

        return executable_uuids, executable_config_lists

    def _compile_grad_buffer_allocations(self, executable_config_lists):
        """Compile gradient buffer allocations."""
        num_mesh = len(self.mesh_group)
        mesh_grad_vars = [{} for _ in range(num_mesh)]
        instruction_lists = defaultdict(
            list)  # Dict[worker -> List[PipelineInstruction]]
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

                # TODO(yonghao): Some var has non-gradient intermediate states
                # that need accumulation. for these vars, we need to record its
                # first mb index when accum will take place.
                grad_uuids[mesh_idx] = self._compile_alloc(
                    grad_vars, grad_sharding_specs, mesh_idx,
                    self.schedule.first_backward_batch_index,
                    global_config.use_memzero_for_gradient_accumulation,
                    instruction_lists, executable_config_lists)

        return grad_uuids, instruction_lists

    def _compile_collect_mesh_input(self, mesh_idx, batch_vars):
        mesh_arg_set = OrderedSet()
        var_to_spec = {}
        mesh_batch_vars = OrderedSet()
        num_batch = self.num_batch
        mesh_arg_indices = []
        input_shard_indices = []
        input_shard_specs = []
        mesh_invar_is_batch = []
        for stage_idx in self.schedule.mesh_stage_mapping[mesh_idx]:
            stage = self.stages[stage_idx]
            for spec, invar in zip(stage.input_sharding_specs, stage.invars):
                if invar in self.global_invars:
                    var_to_spec[invar] = spec
                    if invar in batch_vars:
                        # Split batch arg
                        for batch_idx in range(num_batch):
                            mesh_arg_set.add((invar, batch_idx))
                        mesh_batch_vars.add(invar)
                    else:
                        mesh_arg_set.add((invar, 0))
        mesh_arg_list = list(mesh_arg_set)

        for info in mesh_arg_list:
            var, batch_idx = info
            if batch_idx != 0:
                continue

            global_idx = self.global_invars.index(var)
            mesh_arg_indices.append(global_idx)
            mesh_invar_is_batch.append(self.is_batch[global_idx])

            if self.is_batch[global_idx]:
                aval = var.aval
                batch_dim = 0
                new_shape = (num_batch * aval.shape[0],) + aval.shape[1:]
                new_spec = get_microbatch_sharding_spec(var_to_spec[var],
                                                        batch_dim, num_batch)
                input_shard_indices.append(
                    pxla.spec_to_indices(new_shape, new_spec))
                input_shard_specs.append(new_spec)
            else:
                input_shard_indices.append(
                    pxla.spec_to_indices(var.aval.shape, var_to_spec[var]))
                input_shard_specs.append(var_to_spec[var])
        return (mesh_arg_list, mesh_arg_indices, input_shard_indices,
                input_shard_specs, mesh_invar_is_batch)

    def _compile_split_input_to_microbatches(self, global_batch_invar_set):
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
        num_mesh = len(self.mesh_group)
        mesh_arg_lists = [None for _ in range(num_mesh)]

        # Dispatch args to each mesh
        arg_last_use = {}
        donate_invars = []
        mesh_arg_indices = []
        input_shard_indices = []
        input_shard_specs = []
        batch_invars = []
        for mesh_idx in range(num_mesh):
            (mesh_arg_list, arg_indices, shard_indices, shard_specs,
             is_batch) = self._compile_collect_mesh_input(
                 mesh_idx, global_batch_invar_set)

            mesh_arg_lists[mesh_idx] = mesh_arg_list
            delete_after_run = [
                var in donated_invar_set or var in global_batch_invar_set
                for var, _ in mesh_arg_list
            ]
            donate_invars.append(delete_after_run)
            for info in mesh_arg_list:
                var, batch_idx = info
                if batch_idx != 0:
                    continue
                arg_last_use[var] = mesh_idx

            mesh_arg_indices.append(arg_indices)
            input_shard_indices.append(shard_indices)
            input_shard_specs.append(shard_specs)
            batch_invars.append(is_batch)

        delete_after_shard = []
        for mesh_idx in range(num_mesh):
            delete_after_shard.append([
                self.global_invars[idx] in donated_invar_set and
                arg_last_use[self.global_invars[idx]] == mesh_idx
                for idx in mesh_arg_indices[mesh_idx]
            ])

        # Get local uuids for each input
        input_local_uuid_lists = defaultdict(list)
        for mesh_idx, physical_mesh in enumerate(self.mesh_group):
            mesh_arg_list = mesh_arg_lists[mesh_idx]
            num_args = len(mesh_arg_list)
            # shape: (num_args, num_hosts, num_devices_per_host)
            arg_uuids = self._get_next_uuids(
                num_args * physical_mesh.num_devices).reshape(
                    num_args, -1, physical_mesh.num_devices_per_host)
            for arg_idx, info in enumerate(mesh_arg_lists[mesh_idx]):
                var, batch_idx = info
                self.env.set_var_mesh_uuids(var, batch_idx, mesh_idx,
                                            arg_uuids[arg_idx])
                for worker_idx, worker in enumerate(physical_mesh.workers):
                    input_local_uuid_lists[worker].append(arg_uuids[arg_idx,
                                                                    worker_idx])
        input_config = PipeshardInputConfig(
            input_local_uuid_lists=input_local_uuid_lists,
            donate_invars=donate_invars,
            mesh_arg_indices=mesh_arg_indices,
            input_shard_indices=input_shard_indices,
            delete_after_shard=delete_after_shard,
            batch_invars=batch_invars)
        return input_config, input_shard_specs

    def _compile_concate_get_spec(self, to_concate_vars):
        var_to_spec_all_meshes = []
        output_at = defaultdict(OrderedSet)
        num_mesh = len(self.mesh_group)
        for mesh_idx in range(num_mesh):
            var_to_spec = {}
            for stage_idx in self.schedule.mesh_stage_mapping[mesh_idx]:
                stage = self.stages[stage_idx]
                for spec, outvar in zip(stage.output_sharding_specs,
                                        stage.outvars):
                    if outvar in to_concate_vars:
                        var_to_spec[outvar] = spec
                        output_at[outvar].add(mesh_idx)
            var_to_spec_all_meshes.append(var_to_spec)
        return var_to_spec_all_meshes, output_at

    def _compile_concate(self, instruction_lists, executable_config_lists):
        """
        Generate concate instruction for variables used in non-microbatch part,
        but are not reduced. They should be concated.
        """
        batch_dim = 0
        to_concate_vars = set(self.concat_vars_mapping.values())
        to_concate_specs, output_at = self._compile_concate_get_spec(
            to_concate_vars)
        for var in self.concat_vars_mapping:
            src_var = self.concat_vars_mapping[var]
            dst_mesh_to_uuids = self.env.get_var_meshes(
                var, self.schedule.last_backward_batch_index)
            for mesh_idx in output_at[src_var]:
                physical_mesh = self.mesh_group[mesh_idx]
                # Get input and output uuids
                input_args = np.zeros((self.num_batch, *physical_mesh.shape),
                                      dtype=np.int64)
                for batch_idx in range(self.num_batch):
                    input_args[batch_idx] = self.env.get_var_mesh_uuids(
                        src_var, batch_idx, mesh_idx)
                outputs = self._get_next_uuids(
                    physical_mesh.num_devices).reshape(
                        (1, *physical_mesh.shape))
                dst_mesh_to_uuids[mesh_idx] = outputs.reshape(
                    physical_mesh.shape)
                input_args = input_args.transpose([1, 0, 2])
                outputs = outputs.transpose([1, 0, 2])

                # create and run concat executable
                exec_uuid = next_mesh_executable_uuid()
                spec = to_concate_specs[mesh_idx][src_var]
                hlo_proto = compile_concatenate(xb.get_backend("gpu"),
                                                physical_mesh.shape, spec,
                                                self.num_batch, batch_dim,
                                                src_var.aval)
                exec_config = ConcatWorkerExecutableConfig(exec_uuid, hlo_proto)
                kwargs = {
                    "sync_before": False,
                    "sync_after": False,
                }
                for worker_idx, worker in enumerate(physical_mesh.workers):
                    executable_config_lists[worker].append(exec_config)
                    instruction_lists[worker].append(
                        PipelineInstruction.Run(exec_uuid,
                                                input_args[worker_idx],
                                                outputs[worker_idx], kwargs))

    def _compile_collect_outputs(self) -> None:
        """
        Generate output information.

        This function dispatches output information, including local uuid, local indices to global
        indices, and output specs to each mesh.
        """
        num_mesh = len(self.mesh_group)
        output_local_uuid_list = defaultdict(list)  # Dict[worker -> List[uuid]]

        # collect outvar specs
        var_to_spec_all_meshes = []
        global_outvar_set = OrderedSet(self.global_outvars)
        # This is only a patch. It will be deprecated after we move concat into a stage
        reversed_concat = {
            v: k
            for k, v in self.concat_vars_mapping.items()
            if k in global_outvar_set
        }
        output_at = defaultdict(OrderedSet)
        for mesh_idx in range(num_mesh):
            var_to_spec = {}
            for stage_idx in self.schedule.mesh_stage_mapping[mesh_idx]:
                stage = self.stages[stage_idx]
                for spec, outvar in zip(stage.output_sharding_specs,
                                        stage.outvars):
                    if outvar in global_outvar_set:
                        var_to_spec[outvar] = spec
                        output_at[outvar].add(mesh_idx)
                    if outvar in reversed_concat:
                        concat_outvar = reversed_concat[outvar]
                        var_to_spec[concat_outvar] = spec
                        output_at[concat_outvar].add(mesh_idx)
            var_to_spec_all_meshes.append(var_to_spec)
        # assign indices and get specs
        for outvar in self.global_outvars:
            # the apply gradient only writes to microbatch 0
            mesh_to_uuid = self.env.get_var_meshes(
                outvar, self.schedule.last_backward_batch_index)
            mesh_out_indices = {}
            for mesh_idx in output_at[outvar]:
                mesh = self.mesh_group[mesh_idx]
                uuids = mesh_to_uuid[mesh_idx]
                for worker_idx, worker in enumerate(mesh.workers):
                    output_local_uuid_list[worker].append(uuids[worker_idx])
                mesh_out_indices[mesh_idx] = (
                    len(output_local_uuid_list[mesh.workers[0]]) - 1)
                self.output_spec_list[mesh_idx].append(
                    var_to_spec_all_meshes[mesh_idx][outvar])
            self.mesh_output_indices.append(mesh_out_indices)

        return output_local_uuid_list

    def _compile_alloc(self, variables, sharding_specs, mesh_idx, batch_idx,
                       preallocated, instruction_lists,
                       executable_config_lists):
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

        physical_mesh = self.mesh_group[mesh_idx]
        output_uuids = self._get_next_uuids(
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
        for var_idx, var in enumerate(variables):
            self.env.set_var_mesh_uuids(var, batch_idx, mesh_idx,
                                        transposed[var_idx])
        return output_uuids

    def _get_outs_handler(self):
        """Setup outs handlers that assemble RemoteBufs into DistributedArrays."""
        avals = [outvar.aval for outvar in self.global_outvars]
        is_replicated = [
            bool(len(self.outvar_index_to_mesh_index_mapping[i]) > 1)
            for i, _ in enumerate(self.global_outvars)
        ]

        def outs_handler(mesh_group, bufs):
            ret = []
            for i, aval in enumerate(avals):
                if not is_replicated[i]:
                    # construct DistributedArray
                    mesh_idx = self.outvar_index_to_mesh_index_mapping[i][0]
                    device_mesh = mesh_group[mesh_idx]
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
                        meshes.append(mesh_group[mesh_idx])
                        outvar_index_on_mesh = self.mesh_output_indices[i][
                            mesh_idx]
                        spec = self.output_spec_list[mesh_idx][
                            outvar_index_on_mesh]
                        distributed_arrays.append(
                            DistributedArray(device_mesh=mesh_group[mesh_idx],
                                             aval=aval,
                                             sharding_spec=spec,
                                             remote_buffers=bufs[mesh_idx]
                                             [outvar_index_on_mesh],
                                             indices=pxla.spec_to_indices(
                                                 aval.shape, spec)))
                    arr = ReplicatedDistributedArray(meshes, distributed_arrays)
                ret.append(arr)
            return ret

        return outs_handler

    def _compile_load_info(self, mesh_arg_indices, input_shard_specs):
        assert self.in_tree is not None

        # build load_info_arr: flatten global index => LoadInfo object
        load_info_arr = [None] * len(self.is_batch)
        for mesh_idx, physical_mesh in enumerate(self.mesh_group):
            for local_idx, global_idx in enumerate(mesh_arg_indices[mesh_idx]):
                aval, mesh, spec = (self.global_invars[global_idx].aval,
                                    physical_mesh,
                                    input_shard_specs[mesh_idx][local_idx])
                if load_info_arr[global_idx] is None:
                    load_info_arr[global_idx] = LoadInfo([aval], [mesh], [spec])
                else:
                    load_info_arr[global_idx].add_replica(aval, mesh, spec)

        # build load_info_arr
        return tree_unflatten(self.in_tree, load_info_arr)

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
    def _compile_broadcast_resharding_task(
            src_mesh,
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
            instruction_lists[w].append(
                PipelineInstruction.Broadcast(task_uuid, input_uuids,
                                              output_uuids, "broadcast"))

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
