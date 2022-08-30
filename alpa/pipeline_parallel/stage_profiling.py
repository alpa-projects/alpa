"""Functionalities about profiling the stages."""
from abc import ABC, abstractmethod
from collections import namedtuple
import dataclasses
import gc
import logging
from typing import Dict, Sequence

import jax.numpy as jnp
from jax.core import (ClosedJaxpr, Var, gensym)
from jax.interpreters import pxla
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
import numpy as np
import tqdm
import ray
from ray.exceptions import RayActorError
from ray.util import ActorPool

from alpa.device_mesh import (DistributedArray, PhysicalDeviceMesh,
                              VirtualPhysicalMesh, _shard_device_array)
from alpa.global_env import global_config
from alpa.mesh_executable import (PartialGradAccMeshDriverExecutable,
                                  get_grad_sync_channel_ids)
from alpa.mesh_profiling import (ProfilingResultDatabase,
                                 estimate_hlo_module_cost)
from alpa.pipeline_parallel.apply_grad import APPLY_GRAD_MARKER_SUFFIX
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, get_donation_mapping_and_modify,
    merge_marked_jaxprs_with_named_call, merge_unmarked_with_call,
    rearrange_vars)
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CrossMeshCommunicator, SymbolicReshardingTask, CollectiveGroup,
    ReshardingTaskSpec, SymbolicBroadcastReshardingTask)
from alpa.pipeline_parallel.resharding_tensor import VirtualDistributedArray
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               run_spmd_partitioner_pass,
                                               run_backend_compilation,
                                               hlo_sharding_to_sharding_spec)
from alpa.util import (clone_jaxpr, get_shard_shape, jaxpr_to_hlo_module,
                       OrderedSet, retrieve_placement_group,
                       get_num_available_gpus)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INFINITY_N_STAGES = 4096
GB = 1024**3

CompileOutput = namedtuple("CompileOutput", [
    "model_proto", "stage_plan", "input_sharding_protos",
    "output_sharding_proto", "intermediate_proto",
    "apply_grad_input_sharding_protos"
])

CompileConfig = namedtuple("CompileConfig",
                           ["model_proto", "output_acc_grad_indices"])

ProfileConfig = namedtuple(
    "ProfileConfig",
    ["input_avals", "output_avals", "donate_invars", "output_acc_grad_indices"])

ApplyGradConfig = namedtuple("ApplyGradConfig",
                             ["invars", "apply_grad_only_invars"])

StageConfig = namedtuple(
    "StageConfig", ["compile_config", "profile_config", "apply_grad_config"])


class BaseWorkerPoolWrapper(ABC):
    """Basic wrapper of ray's ActorPool."""

    @abstractmethod
    def __init__(self):
        self.actors = None
        self.pool = None
        self.is_shutdown = False

    def submit(self, fn, value):
        """See ray.util.ActorPool.submit."""
        self.pool.submit(fn, value)

    def get_next(self):
        """See ray.util.ActorPool.get_next."""
        return self.pool.get_next()

    def get_next_unordered(self):
        """See ray.util.ActorPool.get_next_unordered."""
        return self.pool.get_next_unordered(
            timeout=global_config.profile_timeout)

    def shutdown(self, force=True):
        """Shut down the worker."""
        for w in self.actors:
            if force:
                ray.kill(w)
            else:
                w.__ray_terminate__.remote()
        gc.collect()
        self.is_shutdown = True

    def __del__(self):
        if not self.is_shutdown:
            self.shutdown()


def get_input_output_sharding_proto(hlo_module, num_devices):
    """Given proto of XlaComputation, return its input and output sharding."""
    if num_devices <= 1:
        return None, None
    hlo_module.infer_spmd_shardings()
    input_shardings = hlo_module.spmd_parameters_shardings()
    output_sharding = hlo_module.spmd_output_sharding()
    input_sharding_protos = [
        x.proto_tuple().SerializeToString() for x in input_shardings
    ]
    output_sharding_proto = output_sharding.proto_tuple().SerializeToString()
    return input_sharding_protos, output_sharding_proto


class CompileWorker:
    """
    A ray actor to compile Jaxpr to HLO Proto using distributed workers.

    To activate the worker, a gpu resource is required.
    """

    def __init__(self):
        self.cnt = 0

    def compile_stage_for_profiling(self, stage_id, config: CompileConfig,
                                    logical_mesh, autosharding_option,
                                    num_micro_batches):
        """
        Compile a single stage with auto sharding for profiling.

        Args:
            stage_id: the index of the input stage.
            config: configs for compilation.
            logical_mesh: the logical mesh for compilation.
            autosharding_option: the global config dictionary for compilation
                setting.
            num_micro_batches: the number of microbatches.

        Returns:
            proto: The proto of compiled executable
            stage_plan: The sharding strategy from auto sharding
        """
        self.cnt += 1

        # Compile with search to get sharding annotations.
        other_kwargs = {
            "logical_mesh": logical_mesh,
            "return_mode": "stages_and_hook",
            "as_option": autosharding_option,
            "num_micro_batches": num_micro_batches,
            "memory_budget_per_device": None,
        }
        try:
            hlo_module = xe.HloModule.from_serialized_hlo_module_proto(
                config.model_proto)
            # pylint: disable=unbalanced-tuple-unpacking
            module_names, modules, hooked_proto, stage_plan = (
                run_auto_sharding_pass(hlo_module, **other_kwargs))
        except RuntimeError as e:
            logger.warning(f"Compilation error (auto-sharding pass) "
                           f"for stage {stage_id} : {e}")
            return stage_id, None

        assert (len(modules) <=
                2), "Can only compile no more than two stages (compute+(apply))"

        # Read input/output shardings

        if len(modules) > 1:
            if module_names[0].endswith(APPLY_GRAD_MARKER_SUFFIX):
                module_names[0], module_names[1] = module_names[
                    1], module_names[0]
                modules[0], modules[1] = modules[1], modules[0]
            assert module_names[1].endswith(APPLY_GRAD_MARKER_SUFFIX)

        acc_grad_module = modules[0]
        (input_sharding_protos,
         output_sharding_proto) = get_input_output_sharding_proto(
             acc_grad_module, logical_mesh.num_devices)

        if len(modules) > 1:
            apply_grad_input_sharding_protos, _ = (
                get_input_output_sharding_proto(modules[1],
                                                logical_mesh.num_devices))
        else:
            apply_grad_input_sharding_protos = None

        # Compile accumulate_grad part to fully optimized
        rewrite_for_grad_acc = len(config.output_acc_grad_indices) > 0
        try:
            hlo_module = run_spmd_partitioner_pass(
                acc_grad_module,
                logical_mesh.num_devices,
                rewrite_for_grad_acc=rewrite_for_grad_acc,
                rewrite_grad_acc_indices=config.output_acc_grad_indices)
        except IndexError as e:
            logger.warning(f"Compilation error (spmd partitioner pass) "
                           f"for stage {stage_id} : {e}")
            return stage_id, None

        optimized_proto = hlo_module.as_serialized_hlo_module_proto()
        return stage_id, CompileOutput(optimized_proto, stage_plan,
                                       input_sharding_protos,
                                       output_sharding_proto, hooked_proto,
                                       apply_grad_input_sharding_protos)

    @staticmethod
    def run_auto_sharding_pass(stage_id, proto, other_kwargs):
        """Run auto-sharding pass on a proto."""
        hlo_module = xe.HloModule.from_serialized_hlo_module_proto(proto)
        assert other_kwargs["return_mode"] == "stages"
        # pylint: disable=unbalanced-tuple-unpacking
        hlo_stage_names, hlo_stages, stage_plan = run_auto_sharding_pass(
            hlo_module, **other_kwargs)
        hlo_stages = [x.as_serialized_hlo_module_proto() for x in hlo_stages]
        return stage_id, (hlo_stage_names, hlo_stages, stage_plan)


class CompileWorkerPool(BaseWorkerPoolWrapper):
    """A pool of CompileWorker for distributed compilation."""

    def __init__(self, num_cpus, debug_mode=False):
        super().__init__()
        worker_cls = ray.remote(num_cpus=1)(CompileWorker)
        self.actors = [worker_cls.remote() for _ in range(num_cpus)]
        self.pool = ActorPool(self.actors)
        self.local_worker = CompileWorker() if debug_mode else None

    def local_get(self, fn, *value):
        """Debug use function.

        This function submits the work to local worker instead of a remote ray
        actor to help with debug.
        """
        return fn(self.local_worker, *value)


class ProfileWorker:
    """A ray actor to profile a HLO Proto on a given mesh.

    It requests gpu resources from ray. When exceptions is catched, it restarts
    the whole mesh.
    """

    def __init__(self, virtual_mesh: VirtualPhysicalMesh):
        self.mesh = virtual_mesh.get_physical_mesh()
        self.virtual_mesh = virtual_mesh

    def _profile_impl(self, stage_id, compiled_output, profile_info,
                      intermediate_size, initial_size):
        """Implementation of profile function.

        The profiler first compile the HLO Proto into Mesh Executable, then
        profiles the executable and computes the maximal number of stages
        following up this stage.

        Args:
            stage_id: the stage id of the proto.
            compiled_output: Compiled HLO Proto, strategy config, input sharding
                spec and output sharding spec.
            profile_info: input avals, output avals, donation mapping and
                indices in outputs for accumulated gradients.
            intermediate_size: Bytes of intermediates for a microbatch.
            initial_size: Bytes of parameters initially stored, but will
                be not used in the profiled computation, e.g. optimizer states.

        Returns:
            stage_id: the input stage id.
            cost (float): the time to run the profiled stage.
            max_stage: maximal number of stages following up this stage.
            debug_info: other profiled outputs for debug use. This includes
                peak memory during the computation, the total available memory,
                the input intermediate size and input initial size.
        """
        avals, out_avals, tot_donation, output_acc_grad_indices = profile_info
        input_shardings = compiled_output.input_sharding_protos
        output_sharding = compiled_output.output_sharding_proto
        donated_invars = (True,) * len(tot_donation) + (False,) * (
            len(avals) - len(tot_donation))
        hlo_module = xc.XlaComputation(
            compiled_output.model_proto).as_hlo_module()
        if input_shardings is not None:
            hlo_module.set_spmd_parameters_shardings(
                [xe.HloSharding(x) for x in input_shardings])
            hlo_module.set_spmd_output_sharding(xe.HloSharding(output_sharding))
        executable = PartialGradAccMeshDriverExecutable(
            self.mesh, hlo_module, compiled_output.stage_plan, avals, out_avals,
            donated_invars, output_acc_grad_indices)

        # Run profiling
        self.mesh.reset_memory_stats()
        peak_memory = executable.get_total_allocation_size()
        available_memory = self.mesh.get_available_memory()
        cost = executable.profile_with_dummy_inputs(skip_grad_sync=True)
        del executable

        if np.mean(cost) == np.inf:
            max_stage = -1
        else:
            max_stage = int((available_memory - peak_memory - initial_size) //
                            max(intermediate_size, 1e-8) - 1)
            max_stage = min(max(-1, max_stage), INFINITY_N_STAGES)

        return stage_id, cost, max_stage, (peak_memory, available_memory,
                                           intermediate_size, initial_size)

    def profile(self, stage_id, compiled_output, profile_info,
                intermediate_size, initial_size):
        """Run profiling on this profile worker.

        If the RayActorError is catched, it retries until profile_maximum_retry
        is reached. Otherwise, it directly returns. In both cases, the mesh
        restarts.
        """
        for _ in range(global_config.profile_maximum_retry):
            try:
                return self._profile_impl(stage_id, compiled_output,
                                          profile_info, intermediate_size,
                                          initial_size)
            except RayActorError as e:
                logger.warning(f"Meet ray actor error in profiling: {e}")
                self.restart(forced=True)
            except RuntimeError as e:
                logger.warning(f"Meet runtime error in profiling: {e}")
                self.restart(forced=True)
                break
            except AssertionError as e:
                logger.warning(f"Meet assertion error in profiling: {e}")
                self.restart(forced=True)
                break
        return stage_id, np.inf, -1, (np.inf, 0, 0, 0)

    def restart(self, forced):
        """Restart the physical mesh."""
        self.mesh.shutdown(forced=forced)
        self.virtual_mesh.launched_physical_mesh = None
        self.mesh = self.virtual_mesh.get_physical_mesh()


class ProfileWorkerPool(BaseWorkerPoolWrapper):
    """A pool of ProfileWorker for distributed profiling."""

    def __init__(self, virtual_meshes, placement_group):
        super().__init__()
        worker_cls = ray.remote(ProfileWorker)
        self.actors = [
            worker_cls.options(placement_group=placement_group).remote(mesh)
            for mesh in virtual_meshes
        ]
        self.pool = ActorPool(self.actors)


class HloCostModelProfileWorker:
    """A ray actor to estimate the cost of HLO Proto based on cost model."""

    def __init__(self, prof_result, num_devices, num_micro_batches):
        self.backend = xb.get_backend("gpu")
        self.prof_result = prof_result
        self.num_devices = num_devices
        self.num_micro_batches = num_micro_batches

    def profile(self, stage_id, compiled_output, profile_info,
                intermediate_size, initial_size):
        """Use cost model to estimate cost on this profile worker."""
        _, _, _, acc_grad_indices = profile_info
        try:
            compiled = run_backend_compilation(
                self.backend,
                compiled_output.model_proto,
                compiled_output.stage_plan,
                self.num_devices,
                bypass_device_assignment_check=True)
        except RuntimeError as e:
            logger.warning(f"Compilation error (backend codegen): {e}")
            return stage_id, np.inf, -1, (0, 0, 0, 0)

        hlo_module = compiled.hlo_modules()[0]
        grad_sync_channel_ids = ""
        if acc_grad_indices:
            grad_sync_channel_ids = get_grad_sync_channel_ids(hlo_module)
        peak_memory = compiled.total_allocation_size()
        available_memory = self.prof_result.available_memory_per_device
        cost = estimate_hlo_module_cost(hlo_module, self.prof_result,
                                        self.num_micro_batches,
                                        grad_sync_channel_ids)
        del compiled

        #with open(f"/home/ubuntu/efs/alpa/benchmark/alpa/tmp/"
        #          f"profile_stage_{stage_id}.hlo", "w") as fout:
        #    fout.write(hlo_module.to_string())

        if np.mean(cost) == np.inf:
            max_stage = -1
        else:
            max_stage = int((available_memory - peak_memory - initial_size) //
                            max(intermediate_size, 1e-8) - 1)
            max_stage = min(max(-1, max_stage), INFINITY_N_STAGES)

        return stage_id, cost, max_stage, (peak_memory, available_memory,
                                           intermediate_size, initial_size)


class HloCostModelProfileWorkerPool(BaseWorkerPoolWrapper):
    """A pool of HloCostModelProfileWorker for distributed profiling.

    Instead of doing real measurements, this class uses a HLO instruction
    cost model to estimate the cost.
    """

    def __init__(self, num_cpus, placement_group, prof_result, mesh_num_devices,
                 num_micro_batches):
        super().__init__()
        num_gpus = get_num_available_gpus(placement_group)
        gpu_per_cpu = 1
        while gpu_per_cpu * num_cpus > num_gpus:
            gpu_per_cpu /= 2
        env_vars = {"XLA_FLAGS": "--xla_gpu_autotune_level=0"}
        worker_cls = ray.remote(num_cpus=0,
                                num_gpus=gpu_per_cpu)(HloCostModelProfileWorker)
        self.actors = [
            worker_cls.options(
                runtime_env={
                    "env_vars": env_vars
                },
                placement_group=placement_group,
            ).remote(prof_result, mesh_num_devices, num_micro_batches)
            for _ in range(num_cpus)
        ]
        self.pool = ActorPool(self.actors)


def compile_all(stages, num_micro_batches, default_as_option):
    """
    Compile all input stages.
    """
    num_cpus = int(
        min(max(ray.available_resources()["CPU"] // 2, 1), len(stages)))

    compile_workers = CompileWorkerPool(num_cpus)
    for stage_id, (_, stage_config, auto_sharding_config,
                   _) in enumerate(stages):
        logical_mesh, autosharding_option_dict = auto_sharding_config
        compile_workers.submit(
            lambda w, v: w.compile_stage_for_profiling.remote(*v),
            (stage_id, stage_config.compile_config, logical_mesh,
             dataclasses.replace(default_as_option, **
                                 autosharding_option_dict), num_micro_batches))

    compiled_outputs = [None] * len(stages)
    for _ in tqdm.tqdm(stages):
        try:
            stage_id, compiled_output = compile_workers.get_next_unordered()
        except TimeoutError:
            logger.warning("Compile worker timeout")
        except RayActorError as e:
            logger.warning(f"A Compile worker died unexpectedly: {e}")
            continue
        compiled_outputs[stage_id] = compiled_output

    compile_workers.shutdown()
    return compiled_outputs


def profile_all(stages, compiled_outputs: Sequence[CompileOutput], meshes,
                num_layers, num_autosharding_configs, num_micro_batches,
                auto_stage_option, mesh_cached_result):
    """Profile all compiled outputs on given meshes.

    This function launches a profile worker pool and submits given tasks.
    """
    # pylint: disable=unused-argument
    compute_cost, max_n_succ_stages, is_profiled = mesh_cached_result

    placement_group = retrieve_placement_group()

    if auto_stage_option.use_hlo_cost_model:
        num_cpus = int(
            min(max(ray.available_resources()["CPU"] // 2, 1), len(stages)))
        mesh_num_devices = meshes[0].num_devices
        prof_database = ProfilingResultDatabase()
        prof_database.load(auto_stage_option.profiling_database_filename)
        prof_result = prof_database.query("default", meshes[0].shape)
        profile_workers = HloCostModelProfileWorkerPool(num_cpus,
                                                        placement_group,
                                                        prof_result,
                                                        mesh_num_devices,
                                                        num_micro_batches)
    else:
        profile_workers = ProfileWorkerPool(meshes, placement_group)

    succ_compile_ct = 0
    for stage_id, (compiled_output,
                   stage) in enumerate(zip(compiled_outputs, stages)):
        if compiled_output is None:
            continue

        config = compiled_output.stage_plan
        hooked_proto = compiled_output.intermediate_proto
        apply_in_shardings = compiled_output.apply_grad_input_sharding_protos
        (start, end, config_idx), stage_config, _, intermediate_vars = stage
        if is_profiled[start, end, config_idx]:
            continue
        intermediate_size = compute_intermediate_size(hooked_proto,
                                                      intermediate_vars,
                                                      config.logical_mesh_shape)
        apply_grad_input_size = compute_apply_grad_invar_size(
            apply_in_shardings, stage_config.apply_grad_config,
            config.logical_mesh_shape)
        profile_workers.submit(
            lambda w, v: w.profile.remote(*v),
            (stage_id, compiled_output, stage_config.profile_config,
             intermediate_size, apply_grad_input_size))
        succ_compile_ct += 1

    pbar = tqdm.tqdm(range(succ_compile_ct))
    for _ in pbar:
        try:
            (stage_id, cost, max_stage,
             debug_info) = profile_workers.get_next_unordered()
        except TimeoutError:
            profile_workers.shutdown(force=True)
            logger.warning("After waiting for too long, "
                           "all profile workers are forcely killed")
            return compute_cost, max_n_succ_stages, is_profiled
        except (RuntimeError, RayActorError):
            profile_workers.shutdown(force=True)
            logger.warning("Meet unexpected error, "
                           "all profile workers are forcely killed")
            return compute_cost, max_n_succ_stages, is_profiled
        ((start, end, config_idx), _, auto_sharding_config,
         _) = stages[stage_id]
        logical_mesh, auto_sharding_dict = auto_sharding_config
        (peak_memory, available_memory, intermediate_size,
         initial_size) = debug_info
        compute_cost[start, end, config_idx] = np.mean(cost)
        max_n_succ_stages[start, end, config_idx] = max_stage
        is_profiled[start, end, config_idx] = 1
        pbar.write(f"cost[{start}, {end}, {config_idx}]"
                   f"={compute_cost[start, end, config_idx]:.3f},"
                   f" max_n_succ_stage={max_stage},"
                   f" Mem: avail={available_memory / GB:.3f}GB,"
                   f" peak={peak_memory / GB:.3f}GB,"
                   f" intermediate={intermediate_size / GB:.3f}GB,"
                   f" init={initial_size / GB:.3f}GB,"
                   f" as_config={(logical_mesh.shape, auto_sharding_dict)}")
    profile_workers.shutdown()
    return compute_cost, max_n_succ_stages, is_profiled


def split_global_use_and_donate(layers: Sequence[JaxPipelineComputation],
                                layer_indices: OrderedSet[int],
                                donation_mapping: Dict[Var, Var],
                                global_outvars: Sequence[Var]):
    """
    Obtains donation_mapping and global_use of each selected layer.

    It picks some layers (no need to be consecutive) and assumes they are on a
    mesh, it then returns `donation_mapping` and `global_use` of each selected
    layer.

    Args:
        layers: all layers
        layer_indices: indices of selected layers, they are assumed to be in
            the same mesh
        donation_mapping: known global donation mapping
        global_outvars: global outvars

    Returns:
        donation_mapping: donation mapping of all picked layers
        global_used: an OrderedSet of outvars used not only in selected layers
        layers: layers rearranged for donate invar
    """
    reversed_donation_mapping = {v: k for k, v in donation_mapping.items()}
    layer_indices = OrderedSet(layer_indices)
    gensym_fn = gensym([layer.closed_jaxpr().jaxpr for layer in layers])
    num_layers = len(layers)
    out_donation_mapping = {}
    out_global_used = OrderedSet()
    used = OrderedSet(global_outvars)
    local_used = OrderedSet()  # limit donation
    new_layers = []
    for idx in reversed(range(num_layers)):
        layer = layers[idx]
        if idx in layer_indices:
            global_used = OrderedSet()
            local_donation, new_layer = get_donation_mapping_and_modify(
                layer, reversed_donation_mapping, gensym_fn)
            for invar in local_donation:
                assert invar not in global_used and invar not in local_used

            global_used = [var for var in new_layer.outvars if var in used]
            out_donation_mapping.update(local_donation)
            out_global_used.update(global_used)
            local_used.update(new_layer.invars)
            new_layers.append(new_layer)
            continue
        used.update(layer.invars)
    new_layers = list(reversed(new_layers))
    return out_donation_mapping, out_global_used, new_layers


def split_sharding_specs(layers: Sequence[JaxPipelineComputation],
                         mixed_jaxpr: ClosedJaxpr, in_sharding_specs,
                         out_sharding_specs):
    """
    Split sharding specs of layers.

    Some intermediate sharding specs are missed,
    but they are not across meshes so this does not matter.
    """
    in_sharding_dict = dict(zip(mixed_jaxpr.jaxpr.invars, in_sharding_specs))
    out_sharding_dict = dict(zip(mixed_jaxpr.jaxpr.outvars, out_sharding_specs))
    layer_in_sharding_specs = []
    layer_out_sharding_specs = []
    for layer in layers:
        layer_in_sharding_specs.append(
            [in_sharding_dict.get(var, None) for var in layer.invars])
        layer_out_sharding_specs.append(
            [out_sharding_dict.get(var, None) for var in layer.outvars])
    return layer_in_sharding_specs, layer_out_sharding_specs


def generate_stage_info(all_layers, selected_indices, donation_mapping,
                        global_outvars, name, insert_hook_after,
                        apply_grad_layers, apply_grad_info):
    """Combine selected layers together for profiling."""
    # TODO(yonghao): clean up code here
    (selected_donation_mapping, used_outside,
     layers) = split_global_use_and_donate(all_layers, selected_indices,
                                           donation_mapping, global_outvars)

    jaxprs = [layer.closed_jaxpr() for layer in layers]

    merged, intermediate_vars = merge_marked_jaxprs_with_named_call(
        jaxprs, used_outside, selected_donation_mapping, name + "_compute",
        insert_hook_after)

    outvars = OrderedSet(merged.jaxpr.outvars)
    is_donated = [
        invar in selected_donation_mapping and
        selected_donation_mapping[invar] in outvars
        for invar in merged.jaxpr.invars
    ]
    donated_invars = [
        invar for d, invar in zip(is_donated, merged.jaxpr.invars) if d
    ]
    donated_outvars = [selected_donation_mapping[v] for v in donated_invars]
    new_invars = rearrange_vars(merged.jaxpr.invars, donated_invars)
    new_outvars = rearrange_vars(merged.jaxpr.outvars, donated_outvars)
    merged = clone_jaxpr(merged, new_invars, new_outvars)
    compute_avals = [var.aval for var in merged.jaxpr.invars]
    compute_out_avals = [var.aval for var in merged.jaxpr.outvars]
    acc_grad_outvars = set(global_outvars)
    output_acc_grad_indices = [
        i for i, var in enumerate(merged.jaxpr.outvars)
        if var in acc_grad_outvars
    ]
    profile_config = ProfileConfig(compute_avals, compute_out_avals,
                                   list(is_donated), output_acc_grad_indices)

    apply_info = ApplyGradConfig(None, None)

    if apply_grad_layers:
        apply_grad_donation, apply_grad_outvars = apply_grad_info
        merged_apply = merge_marked_jaxprs_with_named_call(
            [layer.closed_jaxpr() for layer in apply_grad_layers],
            apply_grad_outvars, apply_grad_donation, name + "_apply")
        apply_only_invars = OrderedSet(merged_apply.jaxpr.invars).difference(
            new_invars).difference(new_outvars)
        apply_info = ApplyGradConfig(merged_apply.jaxpr.invars,
                                     apply_only_invars)
        names = ["merged", "merged_" + APPLY_GRAD_MARKER_SUFFIX]
        all_outvars = OrderedSet(new_outvars).union(merged_apply.jaxpr.outvars)
        all_outvars = list(all_outvars)
        donation_map = dict(apply_grad_donation)
        donation_map.update(selected_donation_mapping)
        merged, is_donated = merge_unmarked_with_call([merged, merged_apply],
                                                      names, all_outvars,
                                                      donation_map)
    else:
        merged, is_donated = merge_unmarked_with_call([merged], ["merged"],
                                                      new_outvars,
                                                      selected_donation_mapping)

    hlo_module = jaxpr_to_hlo_module(name, merged, is_donated)
    proto = hlo_module.as_serialized_hlo_module_proto()
    compile_config = CompileConfig(proto, output_acc_grad_indices)
    stage_config = StageConfig(compile_config, profile_config, apply_info)
    return intermediate_vars, stage_config


def create_collective_group(src_mesh: PhysicalDeviceMesh,
                            dst_mesh: PhysicalDeviceMesh) -> CollectiveGroup:
    """Create a dummy collective group for profiling."""
    cg = CollectiveGroup(
        OrderedSet(src_mesh.device_strs + dst_mesh.device_strs), src_mesh,
        dst_mesh)
    cg.instantiate()
    return cg


def dummy_resharding_send_recv_strategy(spec: ReshardingTaskSpec):
    """Generates a dummy sharding strategy for profiling."""
    src_loads = {src: 0 for src in spec.src.device_mesh.device_strs}
    dst_loads = {dst: 0 for dst in spec.dst.device_mesh.device_strs}
    return (
        CrossMeshCommunicator._generate_send_recv_resharding_strategy_by_loads(  # pylint: disable=protected-access
            spec, src_loads, dst_loads))


def dummy_resharding_broadcast_strategy(spec: ReshardingTaskSpec):
    """Generates a dummy sharding strategy for profiling."""
    src_loads = {src: 0 for src in spec.src.device_mesh.device_strs}
    dst_loads = {dst: 0 for dst in spec.dst.device_mesh.device_strs}
    return (
        CrossMeshCommunicator._generate_broadcast_resharding_strategy_by_loads(  # pylint: disable=protected-access
            spec, src_loads, dst_loads))


# FIXME(Hao): this function is broken by recent updates. Use with caution.
def profile_layer_communication_cost(
        src: JaxPipelineComputation, dst: JaxPipelineComputation,
        src_outvar_sharding_spec, dst_invar_sharding_spec,
        src_mesh: VirtualPhysicalMesh, dst_mesh: VirtualPhysicalMesh,
        collective_group: CollectiveGroup):
    """Profile communication cost for given two stages.

    It ignores the global load balance, but instead only consider the balance of
    the task. However, as the communication is sequential and SPMD, this does
    not hurt much.
    """
    src_outvars = {v: idx for idx, v in enumerate(src.outvars)}

    backup_use_dummy_value = global_config.use_dummy_value_for_benchmarking
    global_config.use_dummy_value_for_benchmarking = True
    tasks = []
    src_phy_mesh = collective_group.src_mesh
    for idx, invar in enumerate(dst.invars):
        if invar in src_outvars:
            out_sharding_spec = src_outvar_sharding_spec[src_outvars[invar]]
            in_sharding_spec = dst_invar_sharding_spec[idx]
            src_array = VirtualDistributedArray(device_mesh=src_mesh,
                                                aval=invar.aval,
                                                sharding_spec=out_sharding_spec)
            dst_array = VirtualDistributedArray(device_mesh=dst_mesh,
                                                aval=invar.aval,
                                                sharding_spec=in_sharding_spec)
            task_spec = ReshardingTaskSpec(src_array, dst_array, [])
            # create resharding strategy, ignore global load balance
            if global_config.resharding_mode == "send_recv":
                strategy = dummy_resharding_send_recv_strategy(task_spec)
            else:
                strategy = dummy_resharding_broadcast_strategy(task_spec)
            task_spec.set_resharding_strategy(strategy)
            # create distributed array as dummy inputs
            input_indices = pxla.spec_to_indices(invar.aval.shape,
                                                 out_sharding_spec)
            remote_ref = _shard_device_array(jnp.zeros_like(invar.aval),
                                             src_phy_mesh, input_indices)
            DistributedArray(src_phy_mesh, invar.aval, in_sharding_spec,
                             remote_ref, input_indices)
            if global_config.resharding_mode == "send_recv":
                task = SymbolicReshardingTask(task_spec, collective_group,
                                              collective_group.src_mesh,
                                              collective_group.dst_mesh)
            else:
                task = SymbolicBroadcastReshardingTask(
                    task_spec, collective_group, collective_group.src_mesh,
                    collective_group.dst_mesh)
            tasks.append(task)

    for task in tasks:
        task.put_send_recv_tasks()
    src_phy_mesh.sync_workers()
    collective_group.dst_mesh.sync_workers()
    results = []
    for task in tasks:
        results.append(task.do_prepared(task.src_array, True))

    tot_cost = sum(max(result) for result in results)

    global_config.use_dummy_value_for_benchmarking = backup_use_dummy_value
    return tot_cost


def _compute_vars_size(sharding_specs, selected_vars, logical_mesh_shape):
    """Compute bytes of selected_vars with given sharding proto and logical
    mesh."""

    def get_byte(shape, dtype):
        return np.prod(shape) * np.dtype(dtype).itemsize

    if len(selected_vars) == 0:
        return 0

    avals = [v.aval for v in selected_vars]
    if np.prod(logical_mesh_shape) == 1:
        return sum(get_byte(aval.shape, aval.dtype) for aval in avals)

    sharded_shapes = [
        get_shard_shape(aval, spec)
        for aval, spec in zip(avals, sharding_specs)
    ]

    return sum(
        get_byte(shape, aval.dtype)
        for shape, aval in zip(sharded_shapes, avals))


def compute_intermediate_size(serialized_proto, intermediate_vars,
                              logical_mesh_shape):
    """Compute bytes of serialized proto."""
    if len(intermediate_vars) == 0:
        return 0

    avals = [v.aval for v in intermediate_vars]
    if np.prod(logical_mesh_shape) == 1:
        sharding_specs = None
    else:
        hlo_sharding = xe.HloSharding(serialized_proto[0])
        if len(avals) == 1:
            sharding_specs = [
                hlo_sharding_to_sharding_spec(hlo_sharding, avals[0],
                                              logical_mesh_shape)
            ]
        else:
            sharding_specs = hlo_sharding_to_sharding_spec(
                hlo_sharding, avals, logical_mesh_shape)
    return _compute_vars_size(sharding_specs, intermediate_vars,
                              logical_mesh_shape)


def compute_apply_grad_invar_size(input_sharding_protos,
                                  config: ApplyGradConfig, logical_mesh_shape):
    """Compute the size of parameters only used in apply gradient period.

    These parameters are never used in compute gradient period but stored on
    the GPU, so they take memory and influence max_n_succ_stages.
    """
    if config.invars is None:
        assert config.apply_grad_only_invars is None
        return 0
    avals = [v.aval for v in config.invars]
    if np.prod(logical_mesh_shape) == 1:
        selected_sharding_specs = None
        ordered_selected_vars = list(config.apply_grad_only_invars)
    else:
        assert len(input_sharding_protos) == len(config.invars)
        sharding_specs = [
            hlo_sharding_to_sharding_spec(xe.HloSharding(sharding_proto), aval,
                                          logical_mesh_shape)
            for sharding_proto, aval in zip(input_sharding_protos, avals)
        ]
        ordered_selected_vars = []
        selected_sharding_specs = []
        for var, spec in zip(config.invars, sharding_specs):
            if var in config.apply_grad_only_invars:
                ordered_selected_vars.append(var)
                selected_sharding_specs.append(spec)
    return _compute_vars_size(selected_sharding_specs, ordered_selected_vars,
                              logical_mesh_shape)
