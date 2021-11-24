from collections import namedtuple
from dataclasses import dataclass
import enum
import jax
import logging
from typing import Any, Dict, Sequence, List, Callable, Union, Optional

from jax.core import Var
from jax.interpreters import pxla
from jax.lib import xla_client, xla_bridge, xla_extension
import jax.numpy as jnp
import numpy as np
import ray.exceptions

from parax.device_mesh import MeshHostWorker, PhysicalDeviceMesh, DistributedArray, ReplicatedDistributedArray
from parax.global_env import global_config
from parax.mesh_executable import (AllocZeroBufferWorkerExecutable,
                                   MemzeroWorkerExecutable,
                                   PartialGradAccMeshWorkerExecutable,
                                   get_grad_sync_channel_ids_with_hint,
                                   next_mesh_executable_uuid, get_uuid_np_array,
                                   next_remote_buffer_uuid, RemoteBufferRef)
from parax.pipeline_parallel.base_runtime import BaseDistributedRuntime
from parax.pipeline_parallel.cross_mesh_resharding import ReshardingTask
from parax.pipeline_parallel.schedules import cached_property, PipelineSchedule
from parax.pipeline_parallel.computation import XlaShardedPipelineComputation
from parax.timer import timers
from parax.util import OrderedSet, get_shard_shape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

timer_names = {
    "overall": "average",
    "compute": "sum",
    "resharding_send": "sum",
    "resharding_recv": "sum"
}


class PipelineInstType(enum.IntEnum):
    RUN = 0
    SEND = 1
    RECV = 2
    FREE = 3


@dataclass
class PipelineInstruction:
    opcode: PipelineInstType
    task_uuid: Optional[int]
    input_uuids: Optional[np.ndarray]
    output_uuids: Optional[np.ndarray]
    opaques: Optional[Dict[str, Any]]
    info: str
    print_uuids: bool = False

    @classmethod
    def Run(cls, task_uuid, input_uuids, output_uuids, kwargs, info=""):
        return cls(opcode=PipelineInstType.RUN,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=output_uuids,
                   opaques={"kwargs": kwargs},
                   info=info)

    @classmethod
    def Send(cls, task_uuid, input_uuids, info=""):
        return cls(opcode=PipelineInstType.SEND,
                   task_uuid=task_uuid,
                   input_uuids=input_uuids,
                   output_uuids=None,
                   opaques=None,
                   info=info)

    @classmethod
    def Recv(cls, task_uuid, output_uuids, set_empty_buffer, info=""):
        return cls(opcode=PipelineInstType.RECV,
                   task_uuid=task_uuid,
                   input_uuids=None,
                   output_uuids=output_uuids,
                   opaques={"set_empty_buffer": set_empty_buffer},
                   info=info)

    @classmethod
    def Free(cls, input_uuids, info=""):
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


def flatten_uuid_set(container):
    # From Sequence[np.ndarray] to OrderedSet of elements in the array
    container = list(container)
    output = OrderedSet()
    for e in container:
        if isinstance(e, np.ndarray) or isinstance(e, list):
            output.update(flatten_uuid_set(e))
            continue
        output.add(e)
    return output


def logging_instructions(instructions):
    for ins_idx, instruction in enumerate(instructions):
        logger.debug(">>> ins_idx {}: op code {}...".format(
            ins_idx, instruction.opcode))


def get_dict(d: Dict[Any, Dict], k) -> Dict:
    return d.setdefault(k, dict())


def recursive_lookup(d, k):
    if k in d:
        return recursive_lookup(d, d[k])
    return k


class DecentralizedDistributedRuntime(BaseDistributedRuntime):
    """
    A decentralized pipeline_parallel runtime.

    This runtime uses the driver to compile and generate static instructions for each
    worker. It sends the instructions to distributed workers and launches the training.
    See the docstring of `BaseDistributedRuntime`.
    """

    def __init__(self,
                 *,
                 pipeline_stages: List[XlaShardedPipelineComputation],
                 global_invars: List[Var],
                 grad_dummy_invars,
                 global_outvars: List[Var],
                 physical_meshes: List[PhysicalDeviceMesh],
                 dependency: np.ndarray,
                 schedule: PipelineSchedule,
                 is_batch: List[bool],
                 num_batch=1):
        super(DecentralizedDistributedRuntime,
              self).__init__(pipeline_stages=pipeline_stages,
                             global_invars=global_invars,
                             grad_dummy_invars=grad_dummy_invars,
                             global_outvars=global_outvars,
                             physical_meshes=physical_meshes,
                             dependency=dependency,
                             schedule=schedule,
                             is_batch=is_batch,
                             num_batch=num_batch)

        self.uuid_counter = 0
        self.instruction_lists = dict()

        # make this the states of this class
        (executable_config_lists, input_local_uuid_list, grad_uuids,
         accumulated_uuid_lists) = self._compile()

        self._worker_executable_uuid_mapping = dict()
        self._executable_uuid_worker_mapping = dict()
        # we create a PipelineMeshWorkerExecutable for each MeshHostWorker
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            mesh_grad_uuids = list(grad_uuids[mesh_idx])
            for worker_idx, worker in enumerate(physical_mesh.workers):
                acc_grad_local_uuids = []
                if len(mesh_grad_uuids):
                    acc_grad_local_uuids = mesh_grad_uuids[worker_idx]
                args = (self.instruction_lists[worker],
                        input_local_uuid_list[worker],
                        self.output_local_uuid_list[worker],
                        executable_config_lists[worker], acc_grad_local_uuids,
                        accumulated_uuid_lists[worker],
                        self.donate_invars[mesh_idx])
                uuid = next_mesh_executable_uuid()
                worker.put_executable.remote(uuid, PipelineMeshWorkerExecutable,
                                             *args)
                self._worker_executable_uuid_mapping[worker] = uuid
                self._executable_uuid_worker_mapping[uuid] = worker

        # for handling input/outputs
        self.outs_handler: Callable = None
        self._setup_outs_handler()

    def get_next_uuids(self, num) -> np.ndarray:
        ret = np.arange(start=self.uuid_counter,
                        stop=self.uuid_counter + num,
                        dtype=np.int64)
        self.uuid_counter += num
        return ret

    def _compile(self):
        """Precompile the stages and generate static instructions for each worker.

        This function takes symbolic passes, allocates uuids of intermediates, and
        creates instruction lists for all intermediates.
        """
        num_mesh = self.num_mesh
        not_batch_invars = OrderedSet([
            var for var, batch in zip(self.global_invars, self.is_batch)
            if not batch
        ])

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

        # TODO(Hao): make the var_at an attribute instead of a ref.
        var_at = dict()
        # Microbatch-unrelated work
        # compile args for tasks
        (executable_config_lists, executable_uuids,
         grad_uuids) = self._compile_task_configs(var_at)
        # mesh_arg_indices

        input_local_uuid_list = self._compile_split_input_to_microbatches(
            not_batch_invars, var_at)

        # Microbatch-related work
        worker_tmp_instructions = dict()
        for mesh in self.physical_meshes:
            for worker in mesh.workers:
                worker_tmp_instructions[worker] = []
        donation_mapping = [dict() for _ in range(num_mesh)]
        worker_to_idx = dict()
        for mesh_idx, mesh in enumerate(self.physical_meshes):
            for worker_idx, worker in enumerate(mesh.workers):
                worker_to_idx[worker] = (mesh_idx, worker_idx)

        for _, sched in enumerate(self.schedule.schedules):
            for worker in worker_tmp_instructions:
                worker_tmp_instructions[worker] = []
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
                        raise NotImplemented(
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
                                                       mesh_idx, var_at,
                                                       executable_config_lists,
                                                       keys, False)
                    # (args, workers, devices)
                    transposed = output_uuids.transpose([1, 0, 2])
                    recv_uuid_list = list(transposed)

                    for invar, recv_uuids in zip(to_reshard_vars,
                                                 recv_uuid_list):
                        var_key, key = get_invar_key(invar, batch_idx)
                        src_idx, src_uuids = list(var_at[key].items())[0]
                        resharding_task = self._resharding_tasks[src_idx][
                            mesh_idx][var_key]
                        resharding_task: ReshardingTask
                        self._compile_resharding_task(src_idx, mesh_idx,
                                                      src_uuids,
                                                      resharding_task,
                                                      recv_uuids)
                        received_keys.add(key)

                # execute
                # allocate uuids for buffers created by RUN
                for outvar in stage.outvars:
                    key = (repr(outvar), batch_idx)
                    # get uuids of this outvar
                    get_dict(var_at, key)[mesh_idx] = self.get_next_uuids(
                        physical_mesh.total_devices).reshape(
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
                                zip(list(input_uuids[idx]),
                                    list(output_uuids[idx])))

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
                received_uuids = [
                    var_at[key].pop(mesh_idx) for key in received_keys
                ]
                for worker_idx, worker in enumerate(physical_mesh.workers):
                    instructions = worker_tmp_instructions[worker]
                    for uuids in received_uuids:
                        instructions.append(
                            PipelineInstruction.Free(uuids[worker_idx]))
            for worker in worker_tmp_instructions:
                self.instruction_lists[worker].extend(
                    worker_tmp_instructions[worker])
        # output info
        self._compile_collect_outputs(var_at)
        # add FREE insts
        accumulated_uuid_lists = dict()
        for worker in self.instruction_lists:
            used_outside = flatten_uuid_set(self.output_local_uuid_list[worker])
            mesh_idx, worker_idx = worker_to_idx[worker]
            accumulated_uuids = list(grad_uuids[mesh_idx])
            if len(accumulated_uuids):
                accumulated_uuids = accumulated_uuids[worker_idx]
            # numpy for (arg, device)
            accumulated_uuids = [[
                recursive_lookup(donation_mapping[mesh_idx], uuid)
                for uuid in list(uuids)
            ]
                                 for uuids in list(accumulated_uuids)]
            donated = set(donation_mapping[mesh_idx].keys())
            used_outside.update(flatten_uuid_set(accumulated_uuids))
            accumulated_uuid_lists[worker] = accumulated_uuids
            self.instruction_lists[worker] = self._compile_free(
                worker, used_outside, donated)

        return (executable_config_lists, input_local_uuid_list, grad_uuids,
                accumulated_uuid_lists)

    def _compile_alloc(self, vars, sharding_specs, mesh_idx, var_at,
                       executable_config_lists, keys, preallocated):
        config_class = (MemZeroWorkerExecutableConfig
                        if preallocated else AllocateZeroWorkerExecutableConfig)
        avals = [var.aval for var in vars]
        sharded_shapes = [
            get_shard_shape(aval, spec)
            for aval, spec in zip(avals, sharding_specs)
        ]
        dtypes = [aval.dtype for aval in avals]
        exec_uuid = next_mesh_executable_uuid()
        config = config_class(exec_uuid, sharded_shapes, dtypes)

        physical_mesh = self.physical_meshes[mesh_idx]
        output_uuids = self.get_next_uuids(
            len(vars) * physical_mesh.total_devices).reshape(
                len(physical_mesh.workers), len(vars), -1)
        for worker_idx, worker in enumerate(physical_mesh.workers):
            executable_config_lists[worker].append(config)
            if preallocated:
                in_uuids = output_uuids[worker_idx]
                out_uuids = []
            else:
                in_uuids = []
                out_uuids = output_uuids[worker_idx]
            self.instruction_lists[worker].append(
                PipelineInstruction.Run(config.exec_uuid,
                                        in_uuids,
                                        out_uuids, {
                                            "sync_before": False,
                                            "sync_after": False
                                        },
                                        info="mem set zero" if preallocated else
                                        "allocate zero for recv"))

        # (args, workers, devices)
        transposed = output_uuids.transpose([1, 0, 2])
        for var_idx in range(len(vars)):
            key = keys[var_idx]
            get_dict(var_at, key)[mesh_idx] = transposed[var_idx]
        return output_uuids

    def _compile_task_configs(self, var_at):
        """
        Assign uuids for each task and prepare configs, as a replacement of MeshDriverExecutable.__init__

        Returns:
            executable_config_lists (Dict[MeshHostWorker, Sequence[ExecutableConfig]]):
                configs of executables put on each mesh
            executable_uuids (Sequence[int]): uuid for each stage's executable
        """
        num_mesh = len(self.physical_meshes)
        executable_config_lists = dict()
        resharding_config_lists = dict()

        executable_uuids = []

        # Each worker has its own instruction list because Resharding is not SPMD
        for physical_mesh in self.physical_meshes:
            for worker in physical_mesh.workers:
                self.instruction_lists[worker] = list()
                executable_config_lists[worker] = list()
                resharding_config_lists[worker] = list()

        # 1. AllocZeroBuffer executables
        mesh_grad_vars = [dict() for _ in range(num_mesh)]
        # TODO(yonghao): replicated code. abstract this part?
        # collect intermediate buffers in each mesh
        for stage_idx, stage in enumerate(self.stages):
            mesh_indices = list(self.schedule.stage_placement(stage_idx))
            assert len(mesh_indices) == 1
            mesh_idx = mesh_indices[0]
            grad_var_spec_dict = mesh_grad_vars[mesh_idx]
            input_specs = stage.input_sharding_specs
            for var_idx, invar in enumerate(stage.invars):
                if invar in self.grad_dummy_invars:
                    if invar in grad_var_spec_dict:
                        raise NotImplemented(
                            f"accumulate {invar} at multiple stages in a mesh")
                    grad_var_spec_dict[invar] = input_specs[var_idx]
        grad_uuids = [[] for _ in range(num_mesh)]
        if len(grad_var_spec_dict):
            for mesh_idx in range(num_mesh):
                grad_var_spec_dict = mesh_grad_vars[mesh_idx]
                grad_vars, grad_sharding_specs = list(
                    zip(*grad_var_spec_dict.items()))

                # TODO(yonghao): below, we start accumulation according to hints provided by schedule;
                #    this is a case that only works for pipeline-parallel training and gradients.
                #    In some model, some var has non-gradient intermediate states that need accumulation.
                #    for these vars, we need to record its first mb index when accum will take place.
                keys = [(repr(var), self.schedule.first_backward_batch_index)
                        for var in grad_vars]
                grad_uuids[mesh_idx] = self._compile_alloc(
                    grad_vars, grad_sharding_specs, mesh_idx, var_at,
                    executable_config_lists, keys, True)

        # 2. PartialGradAccMeshExecutable
        for stage_idx, stage in enumerate(self.stages):
            exec_uuid = next_mesh_executable_uuid()
            executable_uuids.append(exec_uuid)

            mesh_idx = self.schedule.stage_placement(stage_idx)
            assert len(mesh_idx) == 1
            mesh_idx = list(mesh_idx)[0]
            compiled = stage.get_compiled(self.physical_meshes[mesh_idx])
            hlo_module = compiled.hlo_modules()[0]

            # For debug purpose
            if logger.level <= logging.DEBUG:
                name = f"stage_{stage_idx}.hlo"
                hlo_ir = hlo_module.to_string()
                with open(name, "w") as f:
                    f.write(hlo_ir)

            hlo_proto = hlo_module.as_serialized_hlo_module_proto()
            strategy_config = stage.strategy_config
            grad_sync_channel_ids = get_grad_sync_channel_ids_with_hint(
                hlo_module, stage.output_acc_grad_indices)
            for worker in self.physical_meshes[mesh_idx].workers:
                executable_config_lists[worker].append(
                    PartialGradWorkerExecutableConfig(exec_uuid, hlo_proto,
                                                      strategy_config,
                                                      grad_sync_channel_ids))
        return executable_config_lists, executable_uuids, grad_uuids

    def _compile_split_input_to_microbatches(self, not_batch_invars, var_at):
        """
        Split input info e.g. donation into each mesh after expand batch args.
        The expansion is like:
        before:
        a, b, c, d
        after(b, d are batch args and #mb=2):
        a, b0, b1, c, d0, d1
        Returns:
            mesh_arg_indices (Sequence[Sequence[int]]):
                indices[mesh_idx][i] indicates the index of global_invars(expanded)
                of the i-th input for PipelineWorkerExecutable in mesh_idx-th mesh
            donate_invars (Sequence[Sequence[bool]]):
                list[mesh_idx] is the donate_invars of PipelineWorkerExecutable
                in mesh_idx-th mesh
            input_indices (Sequence[Sequence[Tuple[Index, ...]]]):
                list[mesh_idx] is the input_indices of PipelineWorkerExecutable
                in mesh_idx-th mesh. Here the input_indices are for XLA
                to shard_args instead of indices in input list
            input_local_uuid_list (Dict[MeshHostWorker, np.ndarray]):
        """
        donated_invar_set = OrderedSet()
        global_invar_set = OrderedSet(self.global_invars)
        for stage in self.stages:
            for invar, donate in zip(stage.invars, stage.donated_invars):
                if donate and invar in global_invar_set:
                    donated_invar_set.add(invar)
        num_mesh = len(self.physical_meshes)
        global_invar_indices = dict()
        invar_counter = 0
        mesh_arg_lists = [None for _ in range(num_mesh)]
        self.donate_invars = []
        self.input_indices = []
        self.mesh_arg_indices = []
        # expand barch args
        for invar in self.global_invars:
            if invar in not_batch_invars:
                key = invar, 0
                global_invar_indices[key] = invar_counter
                invar_counter += 1
                continue
            for batch_idx in range(self.num_batch):
                key = invar, batch_idx
                global_invar_indices[key] = invar_counter
                invar_counter += 1
        # dispatch args to each mesh
        for mesh_idx in range(num_mesh):
            mesh_arg_set = OrderedSet()
            var_to_spec = dict()
            for stage_idx in self.schedule.mesh_stage_mapping[mesh_idx]:
                stage = self.stages[stage_idx]
                for spec, invar in zip(stage.input_sharding_specs,
                                       stage.invars):
                    if invar in self.global_invars:
                        var_to_spec[invar] = spec
                        if invar in not_batch_invars:
                            mesh_arg_set.add((invar, 0))
                            continue
                        for batch_idx in range(self.num_batch):
                            mesh_arg_set.add((invar, batch_idx))
            mesh_arg_list = list(mesh_arg_set)
            mesh_arg_lists[mesh_idx] = mesh_arg_list

            self.donate_invars.append(
                [key[0] in donated_invar_set for key in mesh_arg_list])
            self.input_indices.append([
                pxla.spec_to_indices(key[0].aval.shape, var_to_spec[key[0]])
                for key in mesh_arg_list
            ])
            self.mesh_arg_indices.append(
                [global_invar_indices[key] for key in mesh_arg_list])
        # get local uuids for each input:
        input_local_uuid_list = dict()
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            mesh_arg_list = mesh_arg_lists[mesh_idx]
            num_args = len(mesh_arg_list)
            # (num_args, num_hosts, num_device)
            arg_uuids = self.get_next_uuids(
                num_args * physical_mesh.total_devices).reshape(
                    num_args, -1, physical_mesh.num_devices_per_host)
            for arg_idx, key in enumerate(mesh_arg_lists[mesh_idx]):
                key = repr(key[0]), key[1]
                get_dict(var_at, key)[mesh_idx] = arg_uuids[arg_idx]
                for worker_idx, worker in enumerate(physical_mesh.workers):
                    input_local_uuid_list.setdefault(worker, list()).append(
                        arg_uuids[arg_idx, worker_idx])
        return input_local_uuid_list

    def _compile_collect_outputs(self, var_at):
        """
        Dispatch output infos including local uuid, local indices to global
        indices and output specs to each mesh .
        Returns:
            output_local_uuid_list (Dict[MeshHostWorker, Sequence[np.ndarray]]):
                output local uuid of each MeshHostWorker
            mesh_output_indices (Sequence[Dict[int, int]]):
                list[outvar_idx][mesh_idx] indicates the index of the output in
                that mesh corresponding to outvar_idx-th global outputs
            output_spec_list (Sequence[Sequence[ShardingSpec]]):
                list[mesh_idx] is the ShardingSpec of all outputs from
                PipelineWorkerExecutable in mesh_idx-th mesh.
        """
        self.output_local_uuid_list = dict()
        num_mesh = len(self.physical_meshes)

        for mesh in self.physical_meshes:
            for worker in mesh.workers:
                self.output_local_uuid_list[worker] = []
        self.mesh_output_indices = []
        self.output_spec_list = [[] for _ in range(num_mesh)]
        # collect outvar specs
        var_to_spec_all_meshes = []
        global_outvar_set = OrderedSet(self.global_outvars)
        for mesh_idx in range(num_mesh):
            var_to_spec = dict()
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
            mesh_out_indices = dict()
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

    def _compile_resharding_task(self, src_mesh_idx, dst_mesh_idx, src_uuids,
                                 resharding_task: ReshardingTask, recv_uuids):
        """
        Add SEND and RECV PipelineInstructions for a ReshardingTask.
        Args:
            src_mesh_idx: mesh index of the src mesh
            dst_mesh_idx: mesh index of the dst mesh
            src_uuids (np.ndarray): uuids of resharded buffer in src mesh
            resharding_task (ReshardingTask): the task to be compiled
            recv_uuids (np.ndarray): uuids of resharded buffer in dst mesh
        """
        src_mesh = self.physical_meshes[src_mesh_idx]
        dst_mesh = self.physical_meshes[dst_mesh_idx]
        num_devices_per_host = dst_mesh.num_devices_per_host
        send_buf_uuids = {worker: list() for worker in src_mesh.workers}
        recv_buf_uuids = {worker: list() for worker in dst_mesh.workers}
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
            self.instruction_lists[w].append(
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
            self.instruction_lists[w].append(
                PipelineInstruction.Recv(task_uuid, output_uuids, False))

    def _compile_free(self, worker, used_outside, donated):
        """Add FREE PipelineInstruction to recycle memory
        """
        instruction_list: Sequence[
            PipelineInstruction] = self.instruction_lists[worker]
        new_list = []
        cannot_free_uuids = OrderedSet(used_outside)
        cannot_free_uuids.update(donated)
        for instruction in reversed(instruction_list):
            # for free instruction, do not free again
            if instruction.input_uuids is None:
                new_list.append(instruction)
                continue
            input_uuids = flatten_uuid_set(list(instruction.input_uuids))
            if not (instruction.opcode == PipelineInstType.FREE):
                unused_uuids = list(input_uuids.difference(cannot_free_uuids))
                if len(unused_uuids):
                    new_list.append(
                        PipelineInstruction.Free(np.array(unused_uuids)))
            cannot_free_uuids.update(input_uuids)
            new_list.append(instruction)
        return list(reversed(new_list))

    def _exec_split_args(self, args, batch_dim=0):
        split_args = []
        for arg_idx, arg in enumerate(args):
            if self.is_batch[arg_idx]:
                for split in jnp.split(arg, self.num_batch, axis=batch_dim):
                    split_args.append(split)
            else:
                split_args.append(arg)
        return split_args

    def run(self, *args, **kwargs):
        """The run function that maps to train_step()."""
        input_bufs: List[Any] = [None for _ in range(self.num_mesh)]
        input_uuids: List[Any] = [None for _ in range(self.num_mesh)]
        output_bufs: List[Any] = [None for _ in range(self.num_mesh)]
        output_uuids: List[Any] = [None for _ in range(self.num_mesh)]

        num_outs = [
            len(self.output_local_uuid_list[mesh.workers[0]])
            for mesh in self.physical_meshes
        ]

        # check if there is OOM
        if global_config.pipeline_runtime_mode == "paper":
            self._check_alive()
        split_args = self._exec_split_args(args)
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            # ray.get(physical_mesh.workers[0].sync.remote())
            # print(f"before shard_args mesh_idx={mesh_idx} allocated:",
            #       ray.get(physical_mesh.workers[0].get_memory_allocated.remote()) / 1024**3, "max_allocated:",
            #       ray.get(physical_mesh.workers[0].get_max_memory_allocated.remote()) / 1024**3)
            mesh_args = [
                split_args[idx] for idx in self.mesh_arg_indices[mesh_idx]
            ]
            input_bufs[mesh_idx] = physical_mesh.shard_args(
                self.input_indices[mesh_idx], self.donate_invars[mesh_idx],
                mesh_args)
            num_hosts = physical_mesh.num_hosts
            num_devices_per_host = physical_mesh.num_devices_per_host
            input_uuids[mesh_idx] = get_uuid_np_array(input_bufs[mesh_idx])\
                .reshape(len(mesh_args), num_hosts, num_devices_per_host) \
                .transpose([1, 0, 2])
            output_uuids[mesh_idx] = next_remote_buffer_uuid(
                num_hosts * num_outs[mesh_idx] * num_devices_per_host).reshape(
                    num_hosts, num_outs[mesh_idx], num_devices_per_host)
            # print(f"after shard_args mesh_idx={mesh_idx} allocated:",
            #       ray.get(physical_mesh.workers[0].get_memory_allocated.remote()) / 1024**3, "max_allocated:",
            #       ray.get(physical_mesh.workers[0].get_max_memory_allocated.remote()) / 1024**3)

        # Execute
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            for i, worker in enumerate(physical_mesh.workers):
                worker.run_executable.remote(
                    self._worker_executable_uuid_mapping[worker],
                    input_uuids[mesh_idx][i], output_uuids[mesh_idx][i],
                    **kwargs)

        # Handle donation
        for mesh_idx in range(len(self.physical_meshes)):
            inputs = input_bufs[mesh_idx]
            for bufs, donate in zip(inputs, self.donate_invars[mesh_idx]):
                if donate:
                    for buf in bufs:
                        buf.set_deleted_on_workers()
        # construct output_bufs first.
        for mesh_idx, physical_mesh in enumerate(self.physical_meshes):
            num_devices_per_host = physical_mesh.num_devices_per_host
            output_uuid_transposed = output_uuids[mesh_idx].transpose([1, 0, 2])
            output_bufs[mesh_idx] = np.empty(
                (num_outs[mesh_idx], physical_mesh.total_devices), dtype=object)
            for i in range(num_outs[mesh_idx]):
                for j in range(physical_mesh.total_devices):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    dtype = self.global_outvars[
                        self.mesh_index_to_outvar_indices_mapping[mesh_idx]
                        [i]].aval.dtype
                    output_bufs[mesh_idx][i][j] = RemoteBufferRef(
                        physical_mesh,
                        host_id,
                        device_id,
                        output_uuid_transposed[i][host_id][device_id],
                        dtype=dtype)
        return self.outs_handler(output_bufs)

    def _setup_outs_handler(self):
        """Setup outs handlers that assemble RemoteBufs into DistributedArrays."""
        avals = [outvar.aval for outvar in self.global_outvars]
        is_replicated = [
            True
            if len(self.outvar_index_to_mesh_index_mapping[i]) > 1 else False
            for i, _ in enumerate(self.global_outvars)
        ]

        def outs_handler(bufs):
            ret = []
            for i, _ in enumerate(avals):
                aval = avals[i]
                if not is_replicated[i]:
                    # construct DA
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
                    # otherwise, construct RDA
                    meshes = []
                    distributed_arrays = []
                    for j, mesh_idx in enumerate(
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
        mapping = dict()
        for i, outvar in enumerate(self.global_outvars):
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
        mapping = dict()
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
        if timer_name not in timer_names:
            raise RuntimeError(
                "Unrecognized timer name for pipeline parallel runtime. "
                "Query timer name from the following: {}.".format(
                    timer_names.keys()))
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
        for name in timer_names:
            for mesh in self.physical_meshes:
                mesh.reset_remote_timer(name)

    def get_total_allocation_size(self):
        # TODO: compute the theoratical total allocation size
        raise NotImplemented

    def get_hlo_text(self):
        """Return the HLO text for all stages."""
        ret = []
        for i in range(len(self.stages)):
            ret.append(self.stages[i].get_hlo_text())
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
        self._destroy_collective_groups()
        if not self.physical_meshes:
            raise RuntimeError("No physical meshes spawned yet in "
                               "the runtime before shutting down.")
        self.reset_benchmark_timers()
        for mesh in self.physical_meshes:
            mesh.shutdown()

    def _exception_shutdown(self):
        """In this shutdown, some actors might have died."""
        # recycle collective group info
        for i in range(self.num_mesh):
            for j in range(self.num_mesh):
                if i < j and self._collective_groups[i][j]:
                    group_name = self._collective_groups[i][j].group_name
                    # TODO(Hao): move this part of recycling to ray.util.collective instead of here.
                    name = "info_" + group_name
                    try:
                        store = ray.get_actor(name)
                        ray.kill(store)
                    except ValueError:
                        pass
        # TODO(Hao): recycle the NCCLUniqueID named actor. Their name is MD5 hashed.
        #            each of them will takes 1 CPU.
        # recycle info actors
        for mesh in self.physical_meshes:
            mesh.shutdown(forced=True)


class PipelineMeshWorkerExecutable:

    def __init__(self, worker: MeshHostWorker, uuid: int,
                 instructions: Sequence[PipelineInstruction],
                 input_local_uuids: Sequence[int],
                 output_local_uuids: Sequence[int], executable_configs,
                 acc_local_uuids: np.ndarray,
                 acc_out_uuids: Sequence[Sequence[int]], donate_invars):
        # Instruction Lists
        self.my_uuid = uuid
        self.instructions = instructions
        self.input_local_uuids = input_local_uuids
        self.output_local_uuids = output_local_uuids
        self.donate_invars = donate_invars

        # Buffer management
        self.worker = worker
        self.global_buffers = worker.buffers

        self.global_executables = worker.executables
        self.global_send_tasks = worker.send_tasks
        self.global_recv_tasks = worker.recv_tasks

        # my related executables
        self._related_exec_uuids = []

        self.acc_grad_buffers = {}
        self.acc_in_uuids = [list(uuids) for uuids in list(acc_local_uuids)]
        self.acc_out_uuids = acc_out_uuids
        # Create tasks
        for task_config in executable_configs:
            self._related_exec_uuids.append(task_config.exec_uuid)
            if isinstance(task_config, PartialGradWorkerExecutableConfig):
                self.worker.put_executable(task_config.exec_uuid,
                                           PartialGradAccMeshWorkerExecutable,
                                           task_config.hlo_proto,
                                           task_config.strategy_config,
                                           task_config.grad_sync_channel_ids)
                continue
            elif isinstance(task_config, MemZeroWorkerExecutableConfig):
                assert len(self.acc_grad_buffers) == 0
                # allocate buffers
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
        # add preallocated buffers for gradient accumulation
        buffers.update(self.acc_grad_buffers)
        # donate invars
        for global_ids, donate in zip(input_global_uuids, self.donate_invars):
            if donate:
                self.worker.delete_buffers(list(global_ids))
        # monkey patch
        self.worker.buffers = buffers

        # Execute
        timers("overall").start(sync_func=self.worker.sync)
        for instruction in self.instructions:
            #print(instruction)
            #print(f"memory_allocated: {self.worker.get_memory_allocated()/1024**3:.3f} GB  "
            #      f"max_memory_allocated: {self.worker.get_max_memory_allocated()/1024**3:.3f} GB")

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
                timers("resharding_recv").suspend()
            elif instruction.opcode == PipelineInstType.FREE:
                self.worker.delete_buffers(instruction.input_uuids)

        for timer_name in ["compute", "resharding_send", "resharding_recv"]:
            if timer_name in timers:
                timers(timer_name).stop()
        timers("overall").stop(sync_func=self.worker.sync)

        # copy to global env
        assert len(self.output_local_uuids) == len(output_global_uuids)
        for local_ids, global_ids in zip(self.output_local_uuids,
                                         output_global_uuids):
            local_ids = list(local_ids)
            global_ids = list(global_ids)
            for local_id, global_id in zip(local_ids, global_ids):
                self.global_buffers[global_id] = buffers[local_id]

        # now acc_grad_buffers are those after grad acc, before apply grad
        # TODO(yonghao): never donate them
        for in_uuids, out_uuids in zip(self.acc_in_uuids, self.acc_out_uuids):
            for in_uuid, out_uuid in zip(in_uuids, out_uuids):
                self.acc_grad_buffers[in_uuid] = buffers[out_uuid]
        # monkey patch
        self.worker.buffers = self.global_buffers
        # Clean the dict
        buffers.clear()
        return True

    def __del__(self):
        self.worker.delete_executable(self.my_uuid)
        for exec_id in self._related_exec_uuids:
            self.worker.delete_executable(exec_id)
