"""Compile pipeline stages to runtime pipeline instructions."""
from collections import namedtuple, defaultdict
from dataclasses import dataclass
import enum
import logging
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, Set

from jax.core import Var
from jax.interpreters import pxla
import numpy as np

from alpa.global_env import global_config
from alpa.device_mesh import (DistributedArray, PhysicalDeviceMeshGroup,
                              ReplicatedDistributedArray)
from alpa.mesh_executable import next_mesh_executable_uuid
from alpa.parallel_plan import PlacementSpec
from alpa.pipeline_parallel.computation import XlaShardedPipelineComputation
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CrossMeshCommunicator, SymbolicBroadcastReshardingTask,
    SymbolicReshardingTask, ReshardingTask)
from alpa.pipeline_parallel.schedules import PipelineSchedule
from alpa.pipeline_parallel.stage_construction import ManualStageOption
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.util import (DisjointDict, OrderedSet, get_shard_shape,
                       get_microbatch_sharding_spec, compile_concatenate)

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
    # Run a broadcast task
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
    def run(cls, task_uuid, input_uuids, output_uuids, kwargs, info=""):  # noqa
        return cls(opcode=PipelineInstType.RUN,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=output_uuids,
                   opaques={"kwargs": kwargs},
                   info=info)

    @classmethod
    def send(cls, task_uuid, input_uuids, info=""):  # noqa
        return cls(opcode=PipelineInstType.SEND,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=None,
                   opaques=None,
                   info=info)

    @classmethod
    def recv(
            cls,  # noqa
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
    def broadcast(
            cls,  # noqa
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
    def free(cls, input_uuids, info=""):  # noqa
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
ConcatWorkerExecutableConfig = namedtuple("ConcatWorkerExecutableConfig",
                                          ["exec_uuid", "hlo"])
PartialGradWorkerExecutableConfig = namedtuple(
    "PartialGradWorkerExecutableConfig",
    ["exec_uuid", "hlo", "stage_plan", "donated_invars"])

ExecutableConfig = Union[AllocateZeroWorkerExecutableConfig,
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
    """Environment for PipelineInstEmitter."""

    def __init__(self, global_invar_set: Set[Var],
                 global_batch_invar_set: Set[Var],
                 grad_dummy_invars: Dict[Var, Var], schedule: PipelineSchedule):
        self.global_invar_set = global_invar_set
        self.global_batch_invar_set = global_batch_invar_set
        self.grad_dummy_invars = grad_dummy_invars
        self.schedule = schedule
        # Dict[var_key -> Dict[mesh_idx -> array_uuid]]
        # The shape of the numpy array is [num_hosts, num_devices_per_host]
        self.env = {}

    def _get_var_key(self, var, batch_idx):
        if (var in self.global_invar_set and
                var not in self.global_batch_invar_set):
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

    def get_var_mesh_uuid(self, var, batch_idx, mesh_idx) -> int:
        key = self._get_var_key(var, batch_idx)
        try:
            return self.env[key][mesh_idx]
        except KeyError as e:
            print(key, var, batch_idx, mesh_idx)
            print(self.env[key])
            raise e

    def get_var_meshes(self, var, batch_idx) -> Dict[int, int]:
        key = self._get_var_key(var, batch_idx)
        return self.env.setdefault(key, {})

    def set_var_mesh_uuid(self, var, batch_idx, mesh_idx, uuid):
        key = self._get_var_key(var, batch_idx)
        self.env.setdefault(key, {})[mesh_idx] = uuid

    def var_at(self, var, batch_idx, mesh_idx) -> bool:
        key = self._get_var_key(var, batch_idx)
        return mesh_idx in self.env.setdefault(key, {})


@dataclass
class PipeshardInputConfig:
    """Configurations of the inputs for a Pipeshard executable."""
    # The local input uuids
    # List[mesh_idx -> List[arg_uuid]]
    input_local_uuid_lists: Sequence[Sequence[int]]
    # Whether the var should be donated
    # List[mesh_idx -> List[bool]]
    donate_invars: Sequence[Sequence[bool]]
    # List[mesh_idx -> List[arg_idx]]
    mesh_arg_indices: Sequence[Sequence[int]]
    # Cached sharding indices for input arguments
    # List[mesh_idx -> List[sharding_indices]].
    input_shard_indices: Sequence[Sequence[Any]]
    # Whether the argument should be deleted after shard
    # List[mesh_idx -> List[bool]]
    delete_after_shard: Sequence[Sequence[bool]]
    # Whether the argument is a batch argument
    # List[mesh_idx -> List[bool]]
    batch_invars: Sequence[Sequence[bool]]


# TODO(yonghao): use worker_idx as the dict's key
@dataclass
class PipeshardConfig:
    """Configurations of a Pipeshard executable."""
    # Executable configs
    instruction_lists: Dict[Any, Sequence[PipelineInstruction]]
    xla_stages: Sequence[XlaShardedPipelineComputation]
    # FIXME(yonghao): share this setting within a mesh
    executable_configs: Dict[Any, Sequence[ExecutableConfig]]
    executable_uuids: Sequence[int]
    schedule: PipelineSchedule
    # Resharding task configs
    device_str_groups: Sequence[Sequence[OrderedSet]]
    allreduce_groups: Tuple[Sequence[int], Var]
    resharding_tasks: Sequence[ReshardingTask]
    # Input configs
    input_config: PipeshardInputConfig
    grad_uuids: Sequence[np.ndarray]
    reduced_var_uuid_lists: Sequence[np.ndarray]
    # Output configs
    output_local_uuid_list: Sequence[Sequence[int]]
    outs_handler: Callable
    # Others (debug info)
    stage_input_shard_specs: Sequence[Sequence[pxla.ShardingSpec]]
    input_placement_specs: Sequence[PlacementSpec]
    output_placement_specs: Sequence[PlacementSpec]
    default_auto_sharding_option: AutoShardingOption
    manual_stage_option: ManualStageOption
    sharding_annotated_hlo_texts: Sequence[str]
    flop_count: int


class PipelineInstEmitter:
    """Pipeline Instruction Emitter."""

    def __init__(self, *, stages: Sequence[XlaShardedPipelineComputation],
                 global_invars: Sequence[Var], grad_dummy_invars: Dict[Var,
                                                                       Var],
                 global_outvars: Sequence[Var], concat_vars_mapping: Dict[Var,
                                                                          Var],
                 mesh_group: PhysicalDeviceMeshGroup,
                 schedule: PipelineSchedule, is_batch: Sequence[bool],
                 num_batch: int,
                 default_auto_sharding_option: AutoShardingOption,
                 manual_stage_option: ManualStageOption, flop_count: int,
                 allreduce_groups: Tuple[Sequence[int], Var]):
        ##### Input arguments #####
        self.stages = stages
        self.global_invars = global_invars
        self.grad_dummy_invars = grad_dummy_invars
        self.concat_vars_mapping = concat_vars_mapping
        self.global_outvars = global_outvars
        self.mesh_group = mesh_group
        self.num_mesh = len(mesh_group)
        self.schedule = schedule
        self.is_batch = is_batch
        self.num_batch = num_batch
        self.default_auto_sharding_option = default_auto_sharding_option
        self.manual_stage_option = manual_stage_option
        self.flop_count = flop_count
        self.sharding_annotated_hlo_texts = [x.get_hlo_text() for x in stages]
        self.allreduce_groups = allreduce_groups

        ##### Internal states #####
        self.uuid_counter = 0  # counter for local buffer uuid
        global_invar_set = OrderedSet(global_invars)
        global_batch_invar_set = OrderedSet(
            v for v, b in zip(global_invars, is_batch) if b)
        self.env = PipelineInstEmitterHelper(global_invar_set,
                                             global_batch_invar_set,
                                             grad_dummy_invars, schedule)
        self._communicator = None
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

    def _compile_sharding_specs(self):
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
                # TODO(yonghao): delay put_resharding_XXXX_task until pipeshard
                #  executable
                if global_config.resharding_mode == "send_recv":
                    self._resharding_tasks[src_mesh_idx][dst_mesh_idx][
                        var] = SymbolicReshardingTask(spec, cg, src_mesh,
                                                      dst_mesh)
                else:
                    self._resharding_tasks[src_mesh_idx][dst_mesh_idx][
                        var] = SymbolicBroadcastReshardingTask(
                            spec, cg, src_mesh, dst_mesh)

    def _gather_resharding_tasks(self):
        """Gather all resharding tasks into a list."""
        tasks = []
        for src_idx in range(self.num_mesh):
            for dst_idx in range(self.num_mesh):
                tasks.extend(self._resharding_tasks[src_idx][dst_idx].values())
        return tasks

    def _establish_nccl_groups(self):
        """
        Identify NCCL groups based on resharding specs but do not instantiate
        them.

        We establish one collective group between two physical meshes, covering
        all the devices in these two meshes that require NCCL communication.

        Returns:
            device_str_groups (List[List[set]]): a num_mesh x num_mesh matrix.
                Only entries at device_str_groups[i][j] (i < j) are filled,
                entries with i > j are None, because (spec[i][j], spec[j][i])
                will share collective groups.
        """
        self._communicator = CrossMeshCommunicator(self.stages, self.schedule)
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

        # Compile resharding tasks
        self._compile_sharding_specs()
        device_str_groups = self._establish_nccl_groups()
        self._compile_resharding_tasks()

        # Compile forward, backward and apply_grad computations
        (executable_uuids,
         executable_config_lists) = self._compile_computation_executables()

        # Compile gradient buffer allocations
        grad_uuids, instruction_lists = self._compile_grad_buffer_allocations(
            executable_config_lists)

        # Split input into micro batches
        (input_config,
         input_shard_specs) = self._compile_split_input_to_microbatches()

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
        output_local_uuid_list, mesh_output_indices, output_spec_list = (
            self._compile_collect_outputs())
        outs_handler, output_placement_specs = self._get_outs_handler(
            mesh_output_indices, output_spec_list)

        # Add gradient accumulation buffer
        reduced_var_uuid_lists = []
        for mesh_idx in range(num_mesh):
            reduced_var_uuids = grad_uuids[mesh_idx]
            reduced_var_uuids = np.array([
                donation_mapping[mesh_idx].recursive_lookup(uuid)
                for uuid in reduced_var_uuids
            ])
            reduced_var_uuid_lists.append(reduced_var_uuids)
        # Insert buffer free instructions
        for worker in instruction_lists:
            mesh_idx, worker_idx = worker_to_idx[worker]
            used_outside = flatten_uuid_set(output_local_uuid_list[mesh_idx])

            donated = set(donation_mapping[mesh_idx].keys())
            used_outside.update(flatten_uuid_set(reduced_var_uuids))
            instruction_lists[worker] = self._compile_free(
                worker, used_outside, donated, instruction_lists)

        # Compile load info
        input_placement_specs = self._compile_input_placement_spec(
            input_config.mesh_arg_indices, input_shard_specs)

        # Keep the input sharding specs based on pipeline stages
        input_shard_specs = [
            self.stages[idx].input_sharding_specs
            for idx in self.schedule.mesh_stage_mapping
        ]

        return PipeshardConfig(
            # Executable configs
            instruction_lists,
            self.stages,
            executable_config_lists,
            executable_uuids,
            self.schedule,
            # Resharding task configs
            device_str_groups,
            self.allreduce_groups,
            self._gather_resharding_tasks(),
            # Input configs
            input_config,
            grad_uuids,
            reduced_var_uuid_lists,
            # Output configs
            output_local_uuid_list,
            outs_handler,
            # Others
            input_shard_specs,
            input_placement_specs,
            output_placement_specs,
            self.default_auto_sharding_option,
            self.manual_stage_option,
            self.sharding_annotated_hlo_texts,
            self.flop_count)

    def _compile_get_vars_from_mesh(self, invars, dst_specs, mesh_idx,
                                    batch_idx, comm_lists, alloc_lists,
                                    executable_config_lists):
        if len(invars) == 0:
            return
        # TODO(yonghao): only compile alloc once, use multiple times
        recv_uuid_list = self._compile_alloc(invars, dst_specs, mesh_idx,
                                             batch_idx, alloc_lists,
                                             executable_config_lists, "recv")

        for invar, recv_uuid in zip(invars, recv_uuid_list):
            var_key = self.env.get_var_with_accumulate(invar, batch_idx)
            src_idx, src_uuid = list(
                self.env.get_var_meshes(invar, batch_idx).items())[0]
            resharding_task = self._resharding_tasks[src_idx][mesh_idx][var_key]
            if global_config.resharding_mode == "send_recv":
                self._compile_resharding_task(src_uuid, resharding_task,
                                              recv_uuid, comm_lists)
            else:
                self._compile_broadcast_resharding_task(
                    self.mesh_group[src_idx], src_uuid, resharding_task,
                    recv_uuid, comm_lists)

    def _compile_exec_one_mesh(self, mesh_idx, task, executable_uuids,
                               donation_mapping, worker_tmp_instructions):
        batch_idx, stage_idx = task
        physical_mesh = self.mesh_group[mesh_idx]
        stage = self.stages[stage_idx]
        for outvar in stage.outvars:
            # get uuids of this outvar
            output_uuid = self._get_next_uuids(1)[0]
            self.env.set_var_mesh_uuid(outvar, batch_idx, mesh_idx, output_uuid)

        exec_uuid = executable_uuids[stage_idx]
        donated_invars = self.stages[stage_idx].donated_invars

        input_uuids = np.zeros((len(stage.invars),), dtype=np.int64)
        output_uuids = np.zeros((len(stage.outvars),), dtype=np.int64)
        for idx, invar in enumerate(stage.invars):
            input_uuids[idx] = self.env.get_var_mesh_uuid(
                invar, batch_idx, mesh_idx)
        for idx, outvar in enumerate(stage.outvars):
            output_uuids[idx] = self.env.get_var_mesh_uuid(
                outvar, batch_idx, mesh_idx)
        for idx in range(len(stage.invars)):
            if donated_invars[idx]:
                donation_mapping[mesh_idx].update(input_uuids[idx],
                                                  output_uuids[idx])

        for worker in physical_mesh.workers:
            kwargs = {
                "skip_grad_sync": self.schedule.should_skip_grad_sync(task),
                "sync_before": False,
                "sync_after": False,
            }

            worker_tmp_instructions[worker].append(
                PipelineInstruction.run(exec_uuid,
                                        input_uuids,
                                        output_uuids,
                                        kwargs,
                                        info=f"stage {stage_idx}"))

    def _compile_exec_one_tick(self, sched, donation_mapping, instruction_lists,
                               executable_uuids, executable_config_lists):
        worker_tmp_instructions = {}
        for mesh in self.mesh_group:
            for worker in mesh.workers:
                worker_tmp_instructions[worker] = []

        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            batch_idx, stage_idx = task
            stage = self.stages[stage_idx]
            # shard_args for intermediates
            to_reshard_vars = []
            reshard_sharding_specs = []
            for invar, spec in zip(stage.invars, stage.input_sharding_specs):
                if self.env.var_at(invar, batch_idx, mesh_idx):
                    # have a copy at the current mesh
                    continue
                # TODO(yonghao): to avoid congestion, maybe sending from the
                # last one (a.k.a. the latest one receiving it) is better, but
                # we have to create the corresponding cross-mesh communication
                # task.
                # if len(self.env.get_var_meshes(invar, batch_idx)) > 1:
                #     raise NotImplementedError(
                #         "Not support resharding replicated")
                var_key = self.env.get_var_with_accumulate(invar, batch_idx)
                src_idx = list(
                    self.env.get_var_meshes(invar, batch_idx).keys())[0]
                resharding = self._resharding_tasks[src_idx][mesh_idx][var_key]
                if resharding.is_local_allgather_task:
                    spec = resharding.task_spec.dst_sharding_spec
                to_reshard_vars.append(invar)
                reshard_sharding_specs.append(spec)
            self._compile_get_vars_from_mesh(to_reshard_vars,
                                             reshard_sharding_specs, mesh_idx,
                                             batch_idx, instruction_lists,
                                             instruction_lists,
                                             executable_config_lists)

            # execute
            self._compile_exec_one_mesh(mesh_idx, task, executable_uuids,
                                        donation_mapping,
                                        worker_tmp_instructions)

        for worker, worker_instruction in worker_tmp_instructions.items():
            instruction_lists[worker].extend(worker_instruction)

    def _compile_computation_executables(self):
        """Compile executables for forward, backward, and apply_grad
        compuations."""
        executable_uuids = []  # List[stage_idx -> executable_uuids]
        executable_config_lists = defaultdict(
            list)  # Dict[worker -> List[ExecutableConfig]]

        for stage_idx, stage in enumerate(self.stages):
            exec_uuid = next_mesh_executable_uuid()
            executable_uuids.append(exec_uuid)

            mesh_idx = self.schedule.stage_placement(stage_idx)
            assert len(mesh_idx) == 1
            mesh_idx = list(mesh_idx)[0]
            hlo = stage.get_spmd_partitioned()
            exec_config = PartialGradWorkerExecutableConfig(
                exec_uuid, hlo, stage.stage_plan, stage.donated_invars)

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
                    self.schedule.first_backward_batch_index, instruction_lists,
                    executable_config_lists, "grad acc")

        return grad_uuids, instruction_lists

    def _compile_collect_mesh_input(self, mesh_idx):
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
                if invar in self.env.global_invar_set:
                    var_to_spec[invar] = spec
                    if invar in self.env.global_batch_invar_set:
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
                input_shard_specs.append(var_to_spec[var])
            else:
                input_shard_indices.append(
                    pxla.spec_to_indices(var.aval.shape, var_to_spec[var]))
                input_shard_specs.append(var_to_spec[var])
        return (mesh_arg_list, mesh_arg_indices, input_shard_indices,
                input_shard_specs, mesh_invar_is_batch)

    def _compile_split_input_to_microbatches(self):
        """
        Split batch arguments into micro batches.

        The split is like:
        before: a, b, c, d
        after (b, d are batch args and #mb=2): a, b0, b1, c, d0, d1
        """
        donated_invar_set = OrderedSet()
        for stage in self.stages:
            for invar, donate in zip(stage.invars, stage.donated_invars):
                if donate and invar in self.env.global_invar_set:
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
             is_batch) = self._compile_collect_mesh_input(mesh_idx)

            mesh_arg_lists[mesh_idx] = mesh_arg_list
            delete_after_run = [
                var in donated_invar_set or
                (var in self.env.global_batch_invar_set and
                 global_config.always_donate_micro_batch_vars)
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
        input_local_uuid_lists = [[] for _ in range(num_mesh)]
        for mesh_idx in range(num_mesh):
            mesh_arg_list = mesh_arg_lists[mesh_idx]
            num_args = len(mesh_arg_list)
            # shape: (num_args, num_hosts, num_devices_per_host)
            if num_args > 0:
                arg_uuids = self._get_next_uuids(num_args)
                for arg_idx, info in enumerate(mesh_arg_lists[mesh_idx]):
                    var, batch_idx = info
                    self.env.set_var_mesh_uuid(var, batch_idx, mesh_idx,
                                               arg_uuids[arg_idx])
                    input_local_uuid_lists[mesh_idx].append(arg_uuids[arg_idx])
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
                input_args = np.zeros((self.num_batch,), dtype=np.int64)
                for batch_idx in range(self.num_batch):
                    input_args[batch_idx] = self.env.get_var_mesh_uuid(
                        src_var, batch_idx, mesh_idx)
                output_uuid = self._get_next_uuids(1)
                dst_mesh_to_uuids[mesh_idx] = output_uuid[0]

                # create and run concat executable
                exec_uuid = next_mesh_executable_uuid()
                spec = to_concate_specs[mesh_idx][src_var]
                hlo = compile_concatenate(physical_mesh.shape, spec,
                                          self.num_batch, batch_dim,
                                          src_var.aval)
                exec_config = ConcatWorkerExecutableConfig(exec_uuid, hlo)
                kwargs = {
                    "sync_before": False,
                    "sync_after": False,
                }
                for worker in physical_mesh.workers:
                    executable_config_lists[worker].append(exec_config)
                    instruction_lists[worker].append(
                        PipelineInstruction.run(exec_uuid, input_args,
                                                output_uuid, kwargs))

    def _compile_collect_outputs(self):
        """
        Generate output information.

        This function dispatches output information, including local uuid, local
        indices to global indices, and output specs to each mesh.
        """
        # List[mesh_idx -> List[uuid]]
        output_local_uuid_list = [[] for _ in range(self.num_mesh)]
        # List[arg_idx -> Dict[mesh_idx -> int]]
        mesh_output_indices = []
        # List[mesh_idx -> List[arg_idx -> sharding_spec]]
        output_spec_list = [[] for _ in range(self.num_mesh)]

        # collect outvar specs
        var_to_spec_all_meshes = []
        global_outvar_set = OrderedSet(self.global_outvars)
        # This is only a patch. It will be deprecated after we move concat into
        # a stage
        reversed_concat = {
            v: k
            for k, v in self.concat_vars_mapping.items()
            if k in global_outvar_set
        }
        output_at = defaultdict(OrderedSet)
        for mesh_idx in range(self.num_mesh):
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
                output_local_uuid_list[mesh_idx].append(mesh_to_uuid[mesh_idx])
                mesh_out_indices[mesh_idx] = (
                    len(output_local_uuid_list[mesh_idx]) - 1)
                output_spec_list[mesh_idx].append(
                    var_to_spec_all_meshes[mesh_idx][outvar])
            mesh_output_indices.append(mesh_out_indices)

        return output_local_uuid_list, mesh_output_indices, output_spec_list

    def _compile_alloc(self, variables, sharding_specs, mesh_idx, batch_idx,
                       instruction_lists, executable_config_lists, debug):
        """Compile an executable which allocates zero buffers.

        The zero buffers are:
        1) gradient accumulation buffers
        2) temp buffers for receiving tensors
        """
        config_class = AllocateZeroWorkerExecutableConfig
        avals = [var.aval for var in variables]
        sharded_shapes = [
            get_shard_shape(aval, spec)
            for aval, spec in zip(avals, sharding_specs)
        ]
        dtypes = [aval.dtype for aval in avals]
        exec_uuid = next_mesh_executable_uuid()
        config = config_class(exec_uuid, sharded_shapes, dtypes)

        physical_mesh = self.mesh_group[mesh_idx]
        output_uuids = self._get_next_uuids(len(variables))
        for worker in physical_mesh.workers:
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
                                        info="allocate zero for " + debug))

        # shape: (#args, num_hosts, num_devices_per_host)
        for var_idx, var in enumerate(variables):
            self.env.set_var_mesh_uuid(var, batch_idx, mesh_idx,
                                       output_uuids[var_idx])
        return output_uuids

    def _get_outs_handler(self, mesh_output_indices, output_spec_list):
        """
        Setup outs handlers that assemble RemoteBufs into DistributedArrays.
        """
        outvar_idx_to_mesh_idx = {}  # Dict[var_idx -> List[mesh_idx]]
        for i, _ in enumerate(self.global_outvars):
            outvar_idx_to_mesh_idx[i] = list(mesh_output_indices[i].keys())

        avals = [outvar.aval for outvar in self.global_outvars]
        is_replicated = [
            bool(len(outvar_idx_to_mesh_idx[i]) > 1)
            for i, _ in enumerate(self.global_outvars)
        ]

        mesh_idx_list = []
        outvar_index_on_mesh_list = []
        spec_list = []
        indices_list = []
        output_placement_specs = []

        # Generate cached info
        for i, aval in enumerate(avals):
            if not is_replicated[i]:
                # for DistributedArray
                mesh_idx = outvar_idx_to_mesh_idx[i][0]
                outvar_index_on_mesh = mesh_output_indices[i][mesh_idx]
                spec = output_spec_list[mesh_idx][outvar_index_on_mesh]
                mesh_idx_list.append(mesh_idx)
                outvar_index_on_mesh_list.append(outvar_index_on_mesh)
                spec_list.append(spec)
                indices_list.append(pxla.spec_to_indices(aval.shape, spec))

                output_placement_specs.append(
                    PlacementSpec(aval, (mesh_idx_list[-1],), (spec_list[-1],)))
            else:
                # for RepliatedDistributedArray
                mesh_idx_list.append([])
                outvar_index_on_mesh_list.append([])
                spec_list.append([])
                indices_list.append([])

                for mesh_idx in outvar_idx_to_mesh_idx[i]:
                    outvar_index_on_mesh = mesh_output_indices[i][mesh_idx]
                    spec = output_spec_list[mesh_idx][outvar_index_on_mesh]

                    mesh_idx_list[-1].append(mesh_idx)
                    outvar_index_on_mesh_list[-1].append(outvar_index_on_mesh)
                    spec_list[-1].append(spec)
                    indices_list[-1].append(
                        pxla.spec_to_indices(aval.shape, spec))
                output_placement_specs.append(
                    PlacementSpec(aval, tuple(mesh_idx_list[-1]),
                                  tuple(spec_list[-1])))

        def outs_handler(mesh_group, refs):
            ret = []
            for i, aval in enumerate(avals):
                if not is_replicated[i]:
                    # construct DistributedArray
                    mesh_idx = mesh_idx_list[i]
                    device_mesh = mesh_group[mesh_idx]
                    arr = DistributedArray(
                        device_mesh=device_mesh,
                        aval=aval,
                        sharding_spec=spec_list[i],
                        remote_ref=refs[mesh_idx][outvar_index_on_mesh_list[i]],
                        indices=indices_list[i])
                else:
                    # construct RepliatedDistributedArray
                    meshes = []
                    distributed_arrays = []
                    for j, mesh_idx in enumerate(mesh_idx_list[i]):
                        outvar_index_on_mesh = outvar_index_on_mesh_list[i][j]
                        spec = spec_list[i][j]
                        meshes.append(mesh_group[mesh_idx])
                        distributed_arrays.append(
                            DistributedArray(
                                device_mesh=mesh_group[mesh_idx],
                                aval=aval,
                                sharding_spec=spec,
                                remote_ref=refs[mesh_idx][outvar_index_on_mesh],
                                indices=indices_list[i][j]))
                    arr = ReplicatedDistributedArray(meshes, distributed_arrays)
                ret.append(arr)
            return ret

        return outs_handler, output_placement_specs

    def _compile_input_placement_spec(self, mesh_arg_indices,
                                      input_shard_specs):
        # build spec_arr: List[flatten global index -> PlacementSpec]
        spec_arr = [None] * len(self.is_batch)
        for mesh_idx, physical_mesh in enumerate(self.mesh_group):
            for local_idx, global_idx in enumerate(mesh_arg_indices[mesh_idx]):
                shard_spec = input_shard_specs[mesh_idx][local_idx]
                if spec_arr[global_idx] is None:
                    spec_arr[global_idx] = PlacementSpec(
                        self.global_invars[global_idx].aval,
                        (physical_mesh.mesh_id,), (shard_spec,))
                else:
                    old_val = spec_arr[global_idx]
                    spec_arr[global_idx] = PlacementSpec(
                        old_val.aval,
                        old_val.mesh_ids + (physical_mesh.mesh_id,),
                        old_val.sharding_specs + (shard_spec,))

        return spec_arr

    # TODO(yonghao): set empty buffer is not compatiable with local allgather
    @staticmethod
    def _compile_resharding_task(src_uuid: int,
                                 resharding_task: SymbolicReshardingTask,
                                 recv_uuid: int,
                                 instruction_lists,
                                 set_empty_buffer=False):
        """
        Compile and generate SEND and RECV PipelineInstructions for a
        ReshardingTask.

        Args:
            src_mesh: the src mesh
            dst_mesh: the dst mesh
            src_uuids: uuids of resharded buffer in src mesh
            resharding_task: the task to be compiled
            recv_uuids: uuids of resharded buffer in dst mesh
            set_empty_buffer: set the empty buffer when recv or not
        """

        # add send tasks for each worker
        for w, task_uuid in resharding_task.send_worker_task_ids.items():
            instruction_lists[w].append(
                PipelineInstruction.send(task_uuid, [src_uuid]))

        # add recv task for each worker
        allgather_uuid = (resharding_task.allgather_uuid
                          if resharding_task.is_local_allgather_task else None)
        for w, task_uuid in resharding_task.recv_worker_task_ids.items():
            instruction_lists[w].append(
                PipelineInstruction.recv(task_uuid, [recv_uuid],
                                         set_empty_buffer, allgather_uuid))

    @staticmethod
    def _compile_broadcast_resharding_task(
            src_mesh, src_uuid: int,
            resharding_task: SymbolicBroadcastReshardingTask, recv_uuid: int,
            instruction_lists):

        # add broadcast-based resharding task for each worker
        for w, task_uuid in resharding_task.broadcast_worker_task_ids.items():
            output_uuid = None
            input_uuid = None
            if w in src_mesh.workers:
                input_uuid = [src_uuid]
            else:
                output_uuid = [recv_uuid]
            instruction_lists[w].append(
                PipelineInstruction.broadcast(task_uuid, input_uuid,
                                              output_uuid, "broadcast"))

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
                        PipelineInstruction.free(np.array(list(unused_uuids))))
            cannot_free_uuids.update(input_uuids)
            new_list.append(instruction)
        return list(reversed(new_list))


class OverlapFriendlyPipelineInstEmitter(PipelineInstEmitter):
    """Pipeline instruction emitter that allocates buffers earlier."""

    def __init__(self, *args, **kwargs):
        outvar_def_order = kwargs.pop("outvar_def_order")
        super().__init__(*args, **kwargs)
        # Based on stage info, generate cross-mesh communication requirements
        # This formulates what send task is required
        # Dict[int, Dict[int, Tuple(List, List)]]
        # src_mesh_idx -> (dst_mesh_idx -> (Vars, Sharding Specs))
        self.stage_send_vars = [[] for _ in range(len(self.stages))]
        self._get_stage_send_vars(outvar_def_order)

    def _get_stage_send_vars(self, outvar_def_order):
        self._compile_sharding_specs()
        var_defined = {}
        var_at_mesh = {}
        global_invar_set = set(self.global_invars)
        # mesh_idx -> set of stage_idx
        for stage_idx, stage in enumerate(self.stages):
            assert len(self.schedule.stage_placement(stage_idx)) == 1
            mesh_idx = list(self.schedule.stage_placement(stage_idx))[0]
            for var_idx, var in enumerate(stage.invars):
                if (var in global_invar_set or var in self.grad_dummy_invars or
                        mesh_idx in var_at_mesh[var]):
                    if str(var) == "cus":
                        print("skip at mesh", stage_idx, mesh_idx)
                    continue
                else:
                    # Currently we use the first mesh, since there is almost no
                    # redundant computation and the first sends earlier. If the
                    # var is required multiple times, then we might need round-
                    # robin to avoid congestion.
                    src_stage_idx = list(var_defined[var])[0]
                    # once the var is received, it is permanent stored. Maybe
                    # we will can an option to config it.
                    var_at_mesh[var].add(mesh_idx)
                    # insert the recv task
                    self.stage_send_vars[src_stage_idx].append(
                        (mesh_idx, var, stage.input_sharding_specs[var_idx]))

            for var in stage.outvars:
                var_defined.setdefault(var, OrderedSet()).add(stage_idx)
                var_at_mesh.setdefault(var, OrderedSet()).add(mesh_idx)
        print(self.stage_send_vars[0])
        # Reorder send and merge
        for stage_idx, stage in enumerate(self.stages):
            send_vars = self.stage_send_vars[stage_idx]
            def_order = {
                k: i for i, k in enumerate(outvar_def_order[stage_idx])
            }
            send_vars = sorted(send_vars, key=lambda x: (def_order[x[1]], x[0]))
            final_send_seq = []
            for recv_stage_idx, v, spec in send_vars:
                if (len(final_send_seq) != 0 and
                    (final_send_seq[-1][0] == recv_stage_idx)):
                    final_send_seq[-1][1].append(v)
                    final_send_seq[-1][2].append(spec)
                else:
                    final_send_seq.append((recv_stage_idx, [v], [spec]))
            self.stage_send_vars[stage_idx] = final_send_seq

    def _compile_exec_one_tick(self, sched, donation_mapping, instruction_lists,
                               executable_uuids, executable_config_lists):
        exec_insts = {}
        comm_insts = {}
        for mesh in self.mesh_group:
            for worker in mesh.workers:
                exec_insts[worker] = []
                comm_insts[worker] = []
        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            # execute
            self._compile_exec_one_mesh(mesh_idx, task, executable_uuids,
                                        donation_mapping, exec_insts)
        # send immediately after the result is created.
        # we use another iteration to launch exec before alloc zero for recv
        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            batch_idx, stage_idx = task
            if len(self.stage_send_vars[stage_idx]) > 0:
                for recv_info in self.stage_send_vars[stage_idx]:
                    (receiver_idx, received_vars,
                     received_sharding_specs) = recv_info
                    self._compile_get_vars_from_mesh(received_vars,
                                                     received_sharding_specs,
                                                     receiver_idx, batch_idx,
                                                     comm_insts,
                                                     instruction_lists,
                                                     executable_config_lists)
        for worker, insts in exec_insts.items():
            instruction_lists[worker].extend(insts)
            instruction_lists[worker].extend(comm_insts[worker])
