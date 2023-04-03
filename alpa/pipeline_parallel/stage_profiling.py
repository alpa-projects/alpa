"""Functionalities about profiling the stages."""
from abc import ABC, abstractmethod
from collections import namedtuple
import dataclasses
from time import time
from datetime import datetime
import gc
import logging
import pickle
from typing import Dict, Sequence, Tuple

import jax.numpy as jnp
from jax.core import (ClosedJaxpr, Var, gensym)
from jax.interpreters import pxla
from jax.lib import (
    xla_bridge as xb,
    xla_extension as xe
)
import numpy as np
import tqdm
import ray
from ray.exceptions import RayActorError
from ray.util import ActorPool

from alpa.device_mesh import (DistributedArray, PhysicalDeviceMesh,
                              VirtualPhysicalMesh, _shard_device_array,
                              get_global_cluster)
from alpa.global_env import global_config
from alpa.mesh_executable import (PartialGradAccMeshDriverExecutable,
                                  get_grad_sync_channel_ids)
from alpa.mesh_profiling import (ProfilingResultDatabase,
                                 estimate_hlo_module_cost)
from alpa.pipeline_parallel.apply_grad import APPLY_GRAD_MARKER_SUFFIX
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, get_local_donation_mapping_and_add_missing_invars,
    merge_marked_jaxprs_with_named_call, merge_unmarked_with_call)
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CrossMeshCommunicator, SymbolicReshardingTask, CollectiveGroup,
    ReshardingTaskSpec, SymbolicBroadcastReshardingTask)
from alpa.pipeline_parallel.layer_stats import eqn_flops
from alpa.pipeline_parallel.resharding_tensor import VirtualDistributedArray
from alpa.shard_parallel.auto_sharding import (AutoShardingOption,
                                               LogicalDeviceMesh,
                                               run_auto_sharding_pass,
                                               run_spmd_partitioner_pass,
                                               run_backend_compilation,
                                               hlo_sharding_to_sharding_spec)
from alpa.timer import timers
from alpa.util import (get_shard_shape, jaxpr_to_hlo, OrderedSet,
                       retrieve_placement_group, get_num_available_gpus,
                       setup_computation_alias)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

last_compute_cost_file_name = None

INFINITY_N_STAGES = 2**20
GB = 1024**3

ModuleCompileOutput = namedtuple(
    "ModuleCompileOutput",
    ["hlo", "input_sharding_protos", "output_sharding_proto"])

CompileOutput = namedtuple("CompileOutput", [
    "acc_grad_module_compile_outputs", "stage_plan",
    "apply_grad_input_sharding_protos"
])

CompileConfig = namedtuple(
    "CompileConfig",
    ["hlo", "names", "module_donate_invars", "module_acc_grad_outvars_indices"])

ModuleProfileConfig = namedtuple("ModuleProfileConfig", [
    "invar_names", "outvar_names", "invar_avals", "outvar_avals",
    "donated_invars", "acc_grad_invars_indices", "acc_grad_outvars_indices"
])

ApplyGradConfig = namedtuple("ApplyGradConfig",
                             ["invars", "apply_grad_only_invars"])

StageConfig = namedtuple("StageConfig", [
    "n_modules", "compile_config", "module_profile_configs", "apply_grad_config"
])


class ModuleProfileResult(
        namedtuple("ModuleProfileResult", [
            "compute_cost", "peak_memory", "temp_buffer_size", "invar_names",
            "outvar_names", "invar_sizes", "outvar_sizes", "donated_invars",
            "acc_grad_invars_indices", "acc_grad_outvars_indices",
            "available_memory"
        ])):
    """Profile result of a module."""

    def __str__(self):
        invar_size = sum(self.invar_sizes)
        outvar_size = sum(self.outvar_sizes)
        return (f"ModuleProfileResult("
                f"compute_cost={self.compute_cost:.3f}, "
                f"peak_memory={self.peak_memory / GB:.3f} GB, "
                f"invar_size={invar_size / GB:.3f} GB, "
                f"outvar_size={outvar_size / GB:.3f} GB, "
                f"temp_buffer_size={self.temp_buffer_size / GB:.3f} GB, "
                f"available_memory={self.available_memory / GB:.3f} GB)")


class StageProfileResult:
    """Profile result of a stage."""

    def __init__(self, n_modules, initial_var_names, initial_var_sizes):
        self.n_modules = n_modules
        self.module_profile_results: Sequence[ModuleProfileResult] = [
            None
        ] * n_modules
        self.available_memory = None
        self.initial_var_names = tuple(initial_var_names)
        self.initial_var_sizes = tuple(initial_var_sizes)

    def fully_profiled(self):
        return all(r is not None for r in self.module_profile_results)

    def is_module_profiled(self, module_idx):
        return self.module_profile_results[module_idx] is not None

    def add_module_profile_result(self, module_idx, result):
        self.module_profile_results[module_idx] = result
        if self.available_memory is None:
            self.available_memory = result.available_memory
        else:
            self.available_memory = min(self.available_memory,
                                        result.available_memory)

    def __str__(self):
        total_initial_var_size = sum(self.initial_var_sizes)
        return (f"StageProfileResult("
                f"available_memory={self.available_memory / GB:.3f} GB, "
                f"initial_var_size={total_initial_var_size / GB:.3f} GB, "
                f"module_profile_results={self.module_profile_results})")


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
        x.to_proto().SerializeToString() for x in input_shardings
    ]
    output_sharding_proto = output_sharding.to_proto().SerializeToString()
    return input_sharding_protos, output_sharding_proto


class CompileWorker:
    """
    A ray actor to compile Jaxpr to HLO Proto using distributed workers.

    To activate the worker, a gpu resource is required.
    """

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
            hlo: The WrappedHlo of the compiled executable for accumulate grad
            stage_plan: The sharding strategy from auto sharding
            input_sharding_protos: The proto of accumulate grad's input sharding
            output_sharding_protos: same as above
            hooked_proto: The proto of variables from forward to backward
        """

        # Compile with search to get sharding annotations.
        other_kwargs = {
            "logical_mesh": logical_mesh,
            "return_mode": "stages",
            "as_option": autosharding_option,
            "num_micro_batches": num_micro_batches,
            "memory_budget_per_device": None,
        }
        try:
            # pylint: disable=unbalanced-tuple-unpacking
            module_names, hlos, stage_plan = (run_auto_sharding_pass(
                config.hlo, **other_kwargs))
        except RuntimeError as e:
            logger.warning(f"Compilation error (auto-sharding pass) "
                           f"for stage {stage_id} : {e}")
            return stage_id, None

        # Read input/output shardings
        hlo_dict = dict(zip(module_names, hlos))

        assert (sum(
            name.endswith(APPLY_GRAD_MARKER_SUFFIX) for name in config.names) <=
                1), ("Only one apply grad module is allowed in a single stage.")

        acc_grad_module_compile_outputs = []
        apply_grad_input_sharding_protos = None

        for module_id, module_name in enumerate(config.names):
            hlo = hlo_dict[module_name]
            setup_computation_alias(hlo, config.module_donate_invars[module_id])
            module = hlo.get_module()
            if module_name.endswith(APPLY_GRAD_MARKER_SUFFIX):
                apply_grad_input_sharding_protos, _ = (
                    get_input_output_sharding_proto(module,
                                                    logical_mesh.num_devices))
            else:
                acc_grad_outvars_indices = (
                    config.module_acc_grad_outvars_indices[module_id])
                rewrite_for_grad_acc = len(acc_grad_outvars_indices) > 0
                (input_sharding_protos,
                 output_sharding_proto) = get_input_output_sharding_proto(
                     module, logical_mesh.num_devices)

                # Compile accumulate_grad part to fully optimized
                try:
                    optimized_hlo = run_spmd_partitioner_pass(
                        hlo,
                        logical_mesh.num_devices,
                        rewrite_for_grad_acc=rewrite_for_grad_acc,
                        rewrite_grad_acc_indices=acc_grad_outvars_indices)
                except IndexError as e:
                    logger.warning(f"Compilation error (spmd partitioner pass) "
                                   f"for stage {stage_id} : {e}")
                    return stage_id, None
                acc_grad_module_compile_outputs.append(
                    ModuleCompileOutput(optimized_hlo, input_sharding_protos,
                                        output_sharding_proto))

        return stage_id, CompileOutput(acc_grad_module_compile_outputs,
                                       stage_plan,
                                       apply_grad_input_sharding_protos)

    @staticmethod
    def run_auto_sharding_pass(stage_id, hlo, other_kwargs):
        """Run auto-sharding pass on a WrappedHlo."""
        assert other_kwargs["return_mode"] == "stages"
        # pylint: disable=unbalanced-tuple-unpacking
        hlo_stage_names, hlo_stages, stage_plan = run_auto_sharding_pass(
            hlo, **other_kwargs)
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
    """A ray actor to profile a WrappedHlo on a given mesh.

    It requests gpu resources from ray. When exceptions is catched, it restarts
    the whole mesh.
    """

    def __init__(self, virtual_mesh: VirtualPhysicalMesh):
        self.mesh = virtual_mesh.get_physical_mesh()
        self.virtual_mesh = virtual_mesh

    def _profile_impl(self, stage_id, compiled_module_output, stage_plan,
                      profile_config):
        """Implementation of profile function.

        The profiler first compile the WrappedHLO into Mesh Executable, then
        profiles the executable and computes the maximal number of stages
        following up this stage.

        Args:
            stage_id: the stage id of the proto.
            compiled_module_output: Compiled WrappedHlo, input sharding,
                spec and output sharding spec.
            stage_plan: The compiled sharding strategy from the auto sharding
                pass.
            profile_config: Profile config of the module.

        Returns:
            stage_id: the input stage id.
            cost (float): the time to run the profiled stage.
            max_stage: maximal number of stages following up this stage.
            debug_info: other profiled outputs for debug use. This includes
                peak memory during the computation, the total available memory,
                the input intermediate size and input initial size.
        """
        input_avals = profile_config.invar_avals
        output_avals = profile_config.outvar_avals
        donated_invars = profile_config.donated_invars
        input_shardings = compiled_module_output.input_sharding_protos
        output_sharding = compiled_module_output.output_sharding_proto
        hlo = compiled_module_output.hlo
        hlo_module = hlo.get_module()
        if input_shardings is not None:
            hlo_module.set_spmd_parameters_shardings(
                [xe.HloSharding(x) for x in input_shardings])
            hlo_module.set_spmd_output_sharding(xe.HloSharding(output_sharding))
        executable = PartialGradAccMeshDriverExecutable(self.mesh, hlo,
                                                        stage_plan, input_avals,
                                                        output_avals,
                                                        donated_invars)

        # Run profiling
        self.mesh.reset_memory_stats()
        peak_memory = executable.get_total_allocation_size()
        available_memory = self.mesh.get_available_memory()
        cost = executable.profile_with_dummy_inputs(skip_grad_sync=True)
        del executable

        return stage_id, cost, peak_memory, available_memory

    def profile(self, stage_id, compiled_output, stage_plan, profile_info):
        """Run profiling on this profile worker.

        If the RayActorError is catched, it retries until profile_maximum_retry
        is reached. Otherwise, it directly returns. In both cases, the mesh
        restarts.
        """
        for _ in range(global_config.profile_maximum_retry):
            try:
                return self._profile_impl(stage_id, compiled_output, stage_plan,
                                          profile_info)
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
        return stage_id, np.inf, np.inf, 0

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
    """A ray actor to estimate the cost of WrappedHLO based on cost model."""

    def __init__(self, prof_result, num_devices, num_micro_batches):
        self.backend = xb.get_backend(global_config.backend)
        self.prof_result = prof_result
        self.num_devices = num_devices
        self.num_micro_batches = num_micro_batches

    def profile(self, stage_id, compiled_module_output, stage_plan,
                profile_config):
        """Use cost model to estimate cost on this profile worker."""
        try:
            compiled = run_backend_compilation(
                self.backend,
                compiled_module_output.hlo,
                stage_plan,
                self.num_devices,
                bypass_device_assignment_check=True)
        except RuntimeError as e:
            logger.warning(f"Compilation error (backend codegen): {e}")
            return stage_id, np.inf, np.inf, 0

        hlo_module = compiled.hlo_modules()[0]
        grad_sync_channel_ids = ""
        if profile_config.acc_grad_outvars_indices:
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

        return stage_id, cost, peak_memory, available_memory


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


def compile_all(stages, num_micro_batches, default_as_option, profile_results):
    """
    Compile all input stages.
    """
    num_cpus = int(
        min(max(ray.available_resources()["CPU"] // 2, 1), len(stages)))

    compile_workers = CompileWorkerPool(num_cpus)
    num_compiled_stages = 0
    for i, (stage_idx, stage_config, auto_sharding_config) in enumerate(stages):
        if (stage_idx in profile_results and
                profile_results[stage_idx].fully_profiled()):
            continue
        logical_mesh, autosharding_option_dict = auto_sharding_config
        compile_workers.submit(
            lambda w, v: w.compile_stage_for_profiling.remote(*v),
            (i, stage_config.compile_config, logical_mesh,
             dataclasses.replace(default_as_option, **
                                 autosharding_option_dict), num_micro_batches))
        num_compiled_stages += 1

    compiled_outputs = [None] * len(stages)
    for _ in tqdm.tqdm(range(num_compiled_stages)):
        try:
            i, compiled_output = compile_workers.get_next_unordered()
        except TimeoutError:
            logger.warning("Compile worker timeout")
            continue
        except RayActorError as e:
            logger.warning(f"A Compile worker died unexpectedly: {e}")
            continue
        compiled_outputs[i] = compiled_output
        stage_idx, stage_config, auto_sharding_config = stages[i]
        logical_mesh_shape = compiled_output.stage_plan.logical_mesh_shape
        apply_in_shardings = compiled_output.apply_grad_input_sharding_protos
        if apply_in_shardings is not None:
            (initial_var_names,
             initial_var_sizes) = compute_apply_grad_invar_size(
                 apply_in_shardings, stage_config.apply_grad_config,
                 logical_mesh_shape)
        else:
            initial_var_names = ()
            initial_var_sizes = ()
        if stage_idx not in profile_results:
            profile_results[stage_idx] = StageProfileResult(
                stage_config.n_modules, initial_var_names, initial_var_sizes)
        else:
            original_initial_size_dict = dict(
                zip(profile_results[stage_idx].initial_var_names,
                    profile_results[stage_idx].initial_var_sizes))
            new_initial_size_dict = dict(
                zip(initial_var_names, initial_var_sizes))
            assert original_initial_size_dict == new_initial_size_dict, (
                f"Initial sizes mismatch between loaded result and newly "
                f"compiled result: {original_initial_size_dict} "
                f"vs {new_initial_size_dict}.")

    compile_workers.shutdown()
    return compiled_outputs


def generate_module_profile_result(raw_result: Tuple,
                                   profile_config: ModuleProfileConfig,
                                   compile_output: ModuleCompileOutput,
                                   logical_mesh_shape: Tuple[int, ...]):
    compute_costs, peak_memory, available_memory = raw_result
    invar_sizes = get_sharded_size_by_proto(
        compile_output.input_sharding_protos, profile_config.invar_avals,
        logical_mesh_shape, False)
    outvar_sizes = get_sharded_size_by_proto(
        [compile_output.output_sharding_proto], profile_config.outvar_avals,
        logical_mesh_shape)
    donate_invar_sizes = [
        size
        for donated, size in zip(profile_config.donated_invars, invar_sizes)
        if donated
    ]
    temp_buffer_size = (peak_memory - sum(invar_sizes) - sum(outvar_sizes) +
                        sum(donate_invar_sizes))

    return ModuleProfileResult(
        compute_cost=np.mean(compute_costs),
        peak_memory=peak_memory,
        temp_buffer_size=temp_buffer_size,
        invar_names=tuple(profile_config.invar_names),
        outvar_names=tuple(profile_config.outvar_names),
        invar_sizes=invar_sizes,
        outvar_sizes=outvar_sizes,
        donated_invars=tuple(profile_config.donated_invars),
        acc_grad_invars_indices=tuple(profile_config.acc_grad_invars_indices),
        acc_grad_outvars_indices=tuple(profile_config.acc_grad_outvars_indices),
        available_memory=available_memory,
    )


def profile_all(stages, compiled_outputs: Sequence[CompileOutput], meshes,
                num_micro_batches, auto_stage_option, profile_results):
    """Profile all compiled outputs on given meshes.

    This function launches a profile worker pool and submits given tasks.
    """
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

    successful_compile_ct = 0
    for i, (compiled_output, stage) in enumerate(zip(compiled_outputs, stages)):
        if compiled_output is None:
            continue
        stage_idx, stage_config, _ = stage

        for module_id, (acc_grad_module, profile_config) in enumerate(
                zip(compiled_output.acc_grad_module_compile_outputs,
                    stage_config.module_profile_configs)):
            if profile_results[stage_idx].is_module_profiled(module_id):
                continue
            profile_workers.submit(lambda w, v: w.profile.remote(*v),
                                   ((i, module_id), acc_grad_module,
                                    compiled_output.stage_plan, profile_config))
            successful_compile_ct += 1

    pbar = tqdm.tqdm(range(successful_compile_ct))
    for _ in pbar:
        try:
            ((i, module_id),
             *module_raw_result) = profile_workers.get_next_unordered()
        except TimeoutError:
            profile_workers.shutdown(force=True)
            logger.warning("After waiting for too long, "
                           "all profile workers are forcely killed")
            return profile_results
        except (RuntimeError, RayActorError):
            profile_workers.shutdown(force=True)
            logger.warning("Meet unexpected error, "
                           "all profile workers are forcely killed")
            return profile_results
        stage_idx, stage_config, _ = stages[i]
        stage_compile_output = compiled_outputs[i]
        module_profile_result = generate_module_profile_result(
            module_raw_result, stage_config.module_profile_configs[module_id],
            stage_compile_output.acc_grad_module_compile_outputs[module_id],
            stage_compile_output.stage_plan.logical_mesh_shape)
        pbar.write(f"result[{stage_idx}, {module_id}] "
                   f"= {module_profile_result}")
        profile_results[stage_idx].add_module_profile_result(
            module_id, module_profile_result)
    profile_workers.shutdown()
    return profile_results


def generate_training_stages_2d(layers,
                                layer_flops_prefix_sum,
                                accumulator_mapping,
                                acc_grad_invars,
                                acc_grad_outvars,
                                apply_grad_layers,
                                apply_grad_global_info,
                                mesh_id,
                                autosharding_configs,
                                mesh_num_devices,
                                cluster_size,
                                stage_imbalance_tolerance=np.inf):
    print("- Generate all stage infos (Jaxpr -> HLO)")
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    indices = list(range(2 * num_layers))
    computation_source_ratio = mesh_num_devices / cluster_size
    is_full_mesh = computation_source_ratio == 1
    tot_flops = layer_flops_prefix_sum[2 * num_layers]
    stages = []
    for start in tqdm.tqdm(range(0, num_layers)):
        for end in tqdm.tqdm(range(start, num_layers), leave=False):
            if is_full_mesh and not (start == 0 and end == num_layers - 1):
                continue
            flops_ratio = (
                layer_flops_prefix_sum[end + 1] - layer_flops_prefix_sum[start]
                + layer_flops_prefix_sum[2 * num_layers - start] -
                layer_flops_prefix_sum[2 * num_layers - end - 1]) / tot_flops
            if (computation_source_ratio > flops_ratio *
                (1 + stage_imbalance_tolerance) or
                    computation_source_ratio < flops_ratio /
                (1 + stage_imbalance_tolerance)):
                continue
            forward_layer_indices = indices[start:end + 1]
            backward_layer_indices = indices[2 * num_layers - end -
                                             1:2 * num_layers - start]
            selected_apply_grad_layers = [
                apply_grad_layers[idx]
                for idx in forward_layer_indices
                if apply_grad_layers[idx] is not None
            ]
            stage_name = f"stage_{start}_{end}"
            stage_config = generate_stage_info(
                layers, [forward_layer_indices, backward_layer_indices],
                accumulator_mapping, acc_grad_invars, acc_grad_outvars,
                stage_name, selected_apply_grad_layers, apply_grad_global_info)
            for config_idx, autosharding_config in enumerate(
                    autosharding_configs):
                if autosharding_config is not None:
                    stage_indices = (start, end, mesh_id, config_idx)
                    stages.append(
                        (stage_indices, stage_config, autosharding_config))
    return stages


def generate_inference_stages_2d(layers,
                                 layer_flops_prefix_sum,
                                 accumulator_mapping,
                                 acc_grad_invars,
                                 acc_grad_outvars,
                                 apply_grad_layers,
                                 apply_grad_global_info,
                                 mesh_id,
                                 autosharding_configs,
                                 mesh_num_devices,
                                 cluster_size,
                                 stage_imbalance_tolerance=np.inf):
    print("- Generate all stage infos (Jaxpr -> HLO)")
    num_layers = len(layers)
    indices = list(range(2 * num_layers))
    computation_source_ratio = mesh_num_devices / cluster_size
    is_full_mesh = computation_source_ratio == 1
    tot_flops = layer_flops_prefix_sum[num_layers]
    stages = []
    for start in tqdm.tqdm(range(0, num_layers)):
        for end in tqdm.tqdm(range(start, num_layers), leave=False):
            if is_full_mesh and not (start == 0 and end == num_layers - 1):
                continue
            flops_ratio = (layer_flops_prefix_sum[end + 1] -
                           layer_flops_prefix_sum[start]) / tot_flops
            if (computation_source_ratio > flops_ratio *
                (1 + stage_imbalance_tolerance) or
                    computation_source_ratio < flops_ratio /
                (1 + stage_imbalance_tolerance)):
                continue
            forward_layer_indices = indices[start:end + 1]
            selected_apply_grad_layers = [
                apply_grad_layers[idx]
                for idx in forward_layer_indices
                if apply_grad_layers[idx] is not None
            ]
            assert len(selected_apply_grad_layers) == 0, (
                "Inference stage should not have apply_grad_layers")
            stage_name = f"stage_{start}_{end}"
            stage_config = generate_stage_info(layers, [forward_layer_indices],
                                               accumulator_mapping,
                                               acc_grad_invars,
                                               acc_grad_outvars, stage_name,
                                               selected_apply_grad_layers,
                                               apply_grad_global_info)
            for config_idx, autosharding_config in enumerate(
                    autosharding_configs):
                if autosharding_config is not None:
                    stage_indices = (start, end, mesh_id, config_idx)
                    stages.append(
                        (stage_indices, stage_config, autosharding_config))
    return stages


def get_merged_stages_memory_stats(
        profile_results: Sequence[StageProfileResult],
        inference_mode: bool = False):
    initial_var_sizes_dict = {}
    for stage_result in profile_results:
        for name, size in zip(stage_result.initial_var_names,
                              stage_result.initial_var_sizes):
            if name not in initial_var_sizes_dict:
                initial_var_sizes_dict[name] = size
            else:
                assert initial_var_sizes_dict[name] == size, (
                    f"Apply grad invar {name} has different size accross "
                    f"different stages: {initial_var_sizes_dict[name]} "
                    f"vs. {size}.")
    initial_size = sum(initial_var_sizes_dict.values())
    peak_memory = 0
    available_memory = min(
        result.available_memory for result in profile_results)
    n_stages = len(profile_results)
    n_modules = profile_results[0].n_modules
    if inference_mode:
        assert n_modules == 1, "Inference mode should only have 1 module."
        module_execution_orders = [list(range(n_stages))]
    else:
        assert n_modules == 2, ("Only support forward and backward modules in "
                                "training mode.")
        module_execution_orders = [
            list(range(n_stages)),
            list(range(n_stages - 1, -1, -1))
        ]
    assert all(result.n_modules == n_modules for result in profile_results)

    # eliminate_time[var] = k means that the variable can be eliminated after
    # stage k.
    last_used_stage_no = {}
    donation_mapping = {}
    reverse_donation_mapping = {}
    acc_grad_invars = OrderedSet()
    acc_grad_outvars = OrderedSet()
    stage_no = n_stages * n_modules
    for module_id, stage_order in reversed(
            list(enumerate(module_execution_orders))):
        for stage_id in reversed(stage_order):
            stage_no -= 1
            module_result = profile_results[stage_id].module_profile_results[
                module_id]
            for invar in module_result.invar_names:
                if invar not in last_used_stage_no:
                    last_used_stage_no[invar] = stage_no
            for i, (invar, donated) in enumerate(
                    zip(module_result.invar_names,
                        module_result.donated_invars)):
                if donated:
                    # Note: here we assume that we always donate the i-th
                    # invar to the i-th outvar. See rearrange_vars function.
                    donation_mapping[invar] = module_result.outvar_names[i]
                    reverse_donation_mapping[
                        module_result.outvar_names[i]] = invar
            for var_id in module_result.acc_grad_invars_indices:
                acc_grad_invars.add(module_result.invar_names[var_id])
            for var_id in module_result.acc_grad_outvars_indices:
                acc_grad_outvars.add(module_result.outvar_names[var_id])

    all_module_invars = []
    for module_id, stage_order in enumerate(module_execution_orders):
        module_invars = {}
        in_module_vars = OrderedSet()
        for stage_id in stage_order:
            module_result = profile_results[stage_id].module_profile_results[
                module_id]
            for invar, size in zip(module_result.invar_names,
                                   module_result.invar_sizes):
                # If the variable is from another module instead of generated
                # with in the module, it cannot be freed within the execution
                # of a single module, but need to be freed after the module
                # finishes.
                if invar in in_module_vars:
                    continue
                if invar in module_invars:
                    module_invars[invar] = max(module_invars[invar], size)
                else:
                    module_invars[invar] = size
            for outvar in module_result.outvar_names:
                in_module_vars.add(outvar)
        all_module_invars.append(module_invars)

    env = {}
    intermediate_size = None
    stage_no = -1
    for module_id, stage_order in enumerate(module_execution_orders):
        module_invars = all_module_invars[module_id]
        env.update(module_invars)
        for stage_id in stage_order:
            stage_no += 1
            module_result = profile_results[stage_id].module_profile_results[
                module_id]
            for invar, size in zip(module_result.invar_names,
                                   module_result.invar_sizes):
                if invar not in env:
                    env[invar] = size
                else:
                    # env[invar] and size might be different because of
                    # different sharding specs. We take the max for
                    # estimation.
                    env[invar] = max(env[invar], size)
            for outvar, size in zip(module_result.outvar_names,
                                    module_result.outvar_sizes):
                assert outvar not in env
                env[outvar] = size
                if outvar in reverse_donation_mapping:
                    assert reverse_donation_mapping[outvar] in env
                    del env[reverse_donation_mapping[outvar]]
            total_env_size = sum(env.values())
            peak_memory = max(peak_memory,
                              total_env_size + module_result.temp_buffer_size)
            # Remove the variables that are no longer used and is generated
            # within the module.
            var_to_be_eliminated = []
            for var in env:
                if (var not in module_invars and var not in acc_grad_invars and
                        var not in acc_grad_outvars and
                    (var not in last_used_stage_no or
                     last_used_stage_no[var] <= stage_no)):
                    var_to_be_eliminated.append(var)
            for var in var_to_be_eliminated:
                del env[var]
        # Remove the variables that are no longer used
        var_to_be_eliminated = []
        for var in env:
            if (var not in acc_grad_invars and var not in acc_grad_outvars and
                (var not in last_used_stage_no or
                 last_used_stage_no[var] <= stage_no)):
                var_to_be_eliminated.append(var)
        for var in var_to_be_eliminated:
            del env[var]

        # Record the variables that are not eliminated at the end of the
        # last forward module.
        if module_id == 0 and not inference_mode:
            intermediate_size = sum(env.values())

    for var in acc_grad_invars:
        if var not in donation_mapping:
            del env[var]

    for var in acc_grad_outvars:
        del env[var]

    assert len(env) == 0, f"Variables {env.keys()} are not eliminated."

    if inference_mode:
        max_stage = None
    else:
        max_stage = int((available_memory - peak_memory - initial_size) //
                        max(intermediate_size, 1e-8) - 1)
        max_stage = min(max(-1, max_stage), INFINITY_N_STAGES)

    return (available_memory, peak_memory, initial_size, intermediate_size,
            max_stage)


def interpret_profile_result_training_2d(
        profile_results: Dict[Tuple[int, ...],
                              StageProfileResult], num_layers: int,
        num_submesh_choices: int, num_autosharding_configs: int):
    all_compute_cost = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf,
        dtype=np.float64)
    all_max_n_succ_stages = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        -1,
        dtype=np.int64)

    for index in np.ndindex(num_layers, num_layers, num_submesh_choices,
                            num_autosharding_configs):
        if index not in profile_results:
            continue
        profile_result = profile_results[index]
        all_compute_cost[index] = sum(
            result.compute_cost
            for result in profile_result.module_profile_results)
        _, _, _, _, all_max_n_succ_stages[index] = (
            get_merged_stages_memory_stats([profile_result]))

    return all_compute_cost, all_max_n_succ_stages


def interpret_profile_result_inference_2d(
        profile_results: Dict[Tuple[int, ...],
                              StageProfileResult], num_layers: int,
        num_submesh_choices: int, num_autosharding_configs: int):
    all_compute_cost = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf,
        dtype=np.float64)
    all_peak_memory = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf,
        dtype=np.float64)

    for index in np.ndindex(num_layers, num_layers, num_submesh_choices,
                            num_autosharding_configs):
        if index not in profile_results:
            continue
        profile_result = profile_results[index]
        assert len(profile_result.module_profile_results) == 1
        all_compute_cost[index] = (
            profile_result.module_profile_results[0].compute_cost)
        all_peak_memory[index] = (
            profile_result.module_profile_results[0].peak_memory)

    return all_compute_cost, all_peak_memory


def generate_training_stages_1d(layers, accumulator_mapping, acc_grad_invars,
                                acc_grad_outvars, apply_grad_layers,
                                apply_grad_global_info, mesh_id,
                                autosharding_configs):
    print("- Generate all stage infos (Jaxpr -> HLO)")
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    stages = []
    for l in tqdm.tqdm(range(0, num_layers)):
        selected_apply_grad_layers = ([] if apply_grad_layers[l] is None else
                                      [apply_grad_layers[l]])
        stage_name = f"stage_{l}"
        stage_config = generate_stage_info(layers, [(l,),
                                                    (2 * num_layers - l - 1,)],
                                           accumulator_mapping, acc_grad_invars,
                                           acc_grad_outvars, stage_name,
                                           selected_apply_grad_layers,
                                           apply_grad_global_info)
        for config_idx, autosharding_config in enumerate(autosharding_configs):
            if autosharding_config is not None:
                stage_indices = (l, mesh_id, config_idx)
                stages.append(
                    (stage_indices, stage_config, autosharding_config))
    return stages


def generate_inference_stages_1d(layers, accumulator_mapping, acc_grad_invars,
                                 acc_grad_outvars, apply_grad_layers,
                                 apply_grad_global_info, mesh_id,
                                 autosharding_configs):
    print("- Generate all stage infos (Jaxpr -> HLO)")
    num_layers = len(layers)
    stages = []
    for l in tqdm.tqdm(range(0, num_layers)):
        selected_apply_grad_layers = ([] if apply_grad_layers[l] is None else
                                      [apply_grad_layers[l]])
        assert len(selected_apply_grad_layers) == 0, (
            "Inference stage should not have apply_grad_layers")
        stage_name = f"stage_{l}"
        stage_config = generate_stage_info(layers, [(l,)], accumulator_mapping,
                                           acc_grad_invars, acc_grad_outvars,
                                           stage_name,
                                           selected_apply_grad_layers,
                                           apply_grad_global_info)
        for config_idx, autosharding_config in enumerate(autosharding_configs):
            if autosharding_config is not None:
                stage_indices = (l, mesh_id, config_idx)
                stages.append(
                    (stage_indices, stage_config, autosharding_config))
    return stages


def interpret_profile_result_training_1d(
        profile_results: Dict[Tuple[int, ...],
                              StageProfileResult], num_layers: int,
        num_submesh_choices: int, num_autosharding_configs: int):
    all_compute_cost = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf,
        dtype=np.float64)
    all_max_n_succ_stages = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        -1,
        dtype=np.int64)

    for start in range(num_layers):
        for end in range(start, num_layers):
            for submesh_choice in range(num_submesh_choices):
                for config_idx in range(num_autosharding_configs):
                    if any(
                        (l, submesh_choice, config_idx) not in profile_results
                            for l in range(start, end + 1)):
                        continue
                    selected_profile_results = [
                        profile_results[(l, submesh_choice, config_idx)]
                        for l in range(start, end + 1)
                    ]
                    all_compute_cost[
                        start, end, submesh_choice, config_idx] = sum(
                            result.compute_cost
                            for profile_result in selected_profile_results
                            for result in profile_result.module_profile_results)
                    (_, _, _, _, all_max_n_succ_stages[start, end,
                                                       submesh_choice,
                                                       config_idx]
                    ) = get_merged_stages_memory_stats(selected_profile_results)
    return all_compute_cost, all_max_n_succ_stages


def interpret_profile_result_inference_1d(
        profile_results: Dict[Tuple[int, ...],
                              StageProfileResult], num_layers: int,
        num_submesh_choices: int, num_autosharding_configs: int):
    all_compute_cost = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf,
        dtype=np.float64)
    all_peak_memory = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf,
        dtype=np.float64)

    for start in range(num_layers):
        for end in range(start, num_layers):
            for submesh_choice in range(num_submesh_choices):
                for config_idx in range(num_autosharding_configs):
                    if any(
                        (l, submesh_choice, config_idx) not in profile_results
                            for l in range(start, end + 1)):
                        continue
                    selected_profile_results = [
                        profile_results[(l, submesh_choice, config_idx)]
                        for l in range(start, end + 1)
                    ]
                    for result in selected_profile_results:
                        assert len(result.module_profile_results) == 1
                    all_compute_cost[
                        start, end, submesh_choice, config_idx] = sum(
                            profile_result.module_profile_results[0].
                            compute_cost
                            for profile_result in selected_profile_results)
                    (available_memory, peak_memory, _, _,
                     _) = get_merged_stages_memory_stats(
                         selected_profile_results, inference_mode=True)
                    if peak_memory > available_memory:
                        all_compute_cost[start, end, submesh_choice,
                                         config_idx] = np.inf
    return all_compute_cost, all_peak_memory


def distributed_profile_on_mesh(stages, meshes: Sequence[VirtualPhysicalMesh],
                                num_micro_batches, default_as_option,
                                auto_stage_option, profile_results):
    timers("stage-construction-compilation").start()

    if len(stages) == 0:
        # Suspend timers
        timers("stage-construction-compilation").stop()
        return profile_results

    print("- Compile all stages")
    try:
        compiled_outputs = compile_all(stages, num_micro_batches,
                                       default_as_option, profile_results)
    except RayActorError as e:
        logger.warning(f"Compilation fatal error: {e}")
        timers("stage-construction-compilation").stop()
        return profile_results
    timers("stage-construction-compilation").stop()

    print("- Profile all stages")
    # shape of compute_cost and max_n_succ_stages:
    # (num_layers, num_layers, num_autosharding_configs)
    timers("stage-construction-profiling").start()
    profile_results = profile_all(stages, compiled_outputs, meshes,
                                  num_micro_batches, auto_stage_option,
                                  profile_results)
    timers("stage-construction-profiling").stop()
    return profile_results


def check_profile_results_consistent(stages,
                                     profile_results: Dict[Tuple,
                                                           StageProfileResult]):
    for stage_idx, stage_config, _ in stages:
        if stage_idx not in profile_results:
            continue
        profile_result = profile_results[stage_idx]
        assert profile_result.n_modules == stage_config.n_modules
        for module_profile_result, module_profile_config in (
                profile_result.module_profile_results,
                stage_config.module_profile_configs):
            if module_profile_result is None:
                continue
            assert (module_profile_result.invar_names ==
                    module_profile_config.invar_names)
            assert (module_profile_result.outvar_names ==
                    module_profile_config.outvar_names)
            assert (module_profile_result.donated_invars ==
                    module_profile_config.donated_invars)
            assert (module_profile_result.required_outvars_indices ==
                    module_profile_config.required_outvars_indices)


def _get_layer_flops_prefix_sum(layers):
    layer_flops_prefix_sum = [0]
    for layer in layers:
        layer_flops = sum(eqn_flops(eqn) for eqn in layer.eqns)
        layer_flops_prefix_sum.append(layer_flops_prefix_sum[-1] + layer_flops)
    return layer_flops_prefix_sum


def get_compute_cost(
        virtual_mesh: VirtualPhysicalMesh,
        submesh_choices: Sequence[Tuple[int]],
        autosharding_configs: Sequence[Sequence[Tuple[LogicalDeviceMesh,
                                                      dict]]],
        layers: Sequence[JaxPipelineComputation],
        accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var],
        acc_grad_outvars: Sequence[Var],
        apply_grad_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple,
        num_micro_batches: int,
        default_as_option: AutoShardingOption,
        auto_stage_option: "AutoStageOption",
        inference_mode: bool = False):
    """Get computation cost for each possible (stage, mesh) configuration.

    This function enumerates all given submesh choices, then profiles compute
    cost of all stage configuration under the submesh. For each submesh, it
    slices the given mesh or the whole device cluster into submeshes to profile.

    Args:
        virtual_mesh: The whole virtual mesh. If profile_with_whole_ray_cluster
            is turned off in global config, virtual_mesh is sliced into pieces
            to run profiling. Otherwise, the whole device cluster is sliced for
            profiling.
        submesh_choices: All available submesh shape choices.
        autosharding_configs: All auto sharding configs for each submesh.
        layers: Layers for computing and accumulating gradients (forward +
            backward).
        accumulator_mapping: Donation mapping from accumulator to
            accumulated results for all layers.
        acc_grad_outvars: Global input variables for all layers.
        acc_grad_outvars: Global output variables for all layers.
        apply_grad_layers: Apply gradient computations corresponding to each
            forward layers.
        apply_grad_global_info: Donation mapping and outvars for apply gradient
            stages.
        default_as_option: The default auto-sharding options.
        auto_stage_option: The auto stage construction algorthm options.
        inference_mode: Whether to run in inference mode.

    Returns:
        Two np.ndarray, each with shape (L, L, S, C), where L is the number of
        forward layers, S is the number of submesh choices, and C is the maximal
        number of autosharding configs for a submesh choice.
        At index (i, j, s, c), the array stores the value under the condition:
        the stage contains forward layers i, i+1, ... j and corresponding
        backward layers, and runs under the s-th submesh and c-th auto sharding
        config for the submesh.
        compute_cost: The compute cost of all possible configurations.
        max_n_succ_stages: The maximal number of succeeding stages. This
            is calculated based on memory constraints.
    """
    cluster_size = virtual_mesh.num_devices
    layer_flops_prefix_sum = _get_layer_flops_prefix_sum(layers)
    if inference_mode:
        num_layers = len(layers)
    else:
        assert len(layers) % 2 == 0
        num_layers = len(layers) // 2
    num_submesh_choices = len(submesh_choices)
    num_autosharding_configs = len(autosharding_configs[0])

    if auto_stage_option.cached_profile_result is not None:
        with open(auto_stage_option.cached_profile_result, "rb") as f:
            profile_results = pickle.load(f)
    else:
        profile_results = {}
    print("-" * 20 + " Automatic stage clustering " + "-" * 20)
    print(f"submesh_choices: {submesh_choices}")

    # Reverse submesh_choices to test larger meshes first
    for mesh_id, submesh in reversed(list(enumerate(submesh_choices))):
        print(f"- Profiling for submesh {mesh_id} {submesh}:")
        num_hosts, num_devices_per_host = submesh
        tic = time()
        if global_config.profile_with_whole_ray_cluster:
            whole_cluster_virtual_mesh = get_global_cluster(
            ).get_virtual_physical_mesh()
            sliced_virtual_meshes = (
                whole_cluster_virtual_mesh.slice_profiling_submeshes(
                    num_hosts, num_devices_per_host))
        else:
            sliced_virtual_meshes = virtual_mesh.slice_profiling_submeshes(
                num_hosts, num_devices_per_host)

        if auto_stage_option.layer_profile_mode == "composition":
            if inference_mode:
                stages = generate_inference_stages_2d(
                    layers, layer_flops_prefix_sum, accumulator_mapping,
                    acc_grad_invars, acc_grad_outvars, apply_grad_layers,
                    apply_grad_global_info, mesh_id,
                    autosharding_configs[mesh_id],
                    sliced_virtual_meshes[0].num_devices, cluster_size,
                    auto_stage_option.stage_imbalance_tolerance)
            else:
                stages = generate_training_stages_2d(
                    layers, layer_flops_prefix_sum, accumulator_mapping,
                    acc_grad_invars, acc_grad_outvars, apply_grad_layers,
                    apply_grad_global_info, mesh_id,
                    autosharding_configs[mesh_id],
                    sliced_virtual_meshes[0].num_devices, cluster_size,
                    auto_stage_option.stage_imbalance_tolerance)
        elif auto_stage_option.layer_profile_mode == "individual":
            if inference_mode:
                stages = generate_inference_stages_1d(
                    layers, accumulator_mapping, acc_grad_invars,
                    acc_grad_outvars, apply_grad_layers, apply_grad_global_info,
                    mesh_id, autosharding_configs[mesh_id])
            else:
                stages = generate_training_stages_1d(
                    layers, accumulator_mapping, acc_grad_invars,
                    acc_grad_outvars, apply_grad_layers, apply_grad_global_info,
                    mesh_id, autosharding_configs[mesh_id])
        else:
            raise ValueError(f"Unknown layer profile mode: "
                             f"{auto_stage_option.layer_profile_mode}")

        check_profile_results_consistent(stages, profile_results)

        profile_results = distributed_profile_on_mesh(
            stages, sliced_virtual_meshes, num_micro_batches, default_as_option,
            auto_stage_option, profile_results)

        toc = time()
        print(f"Profiling for submesh {mesh_id} {submesh} takes {toc - tic:.2f}"
              f" seconds")
        print("-" * 50)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    profile_result_file_name = (f"profile-results-{timestamp}.npy")
    np.save(profile_result_file_name, profile_results)
    global last_compute_cost_file_name
    last_compute_cost_file_name = profile_result_file_name
    print(f"Profile result saved to: {profile_result_file_name}")
    print("-" * 70)

    if auto_stage_option.layer_profile_mode == "composition":
        if inference_mode:
            compute_cost, _ = interpret_profile_result_inference_2d(
                profile_results, num_layers, num_submesh_choices,
                num_autosharding_configs)
            max_n_succ_stages = None
        else:
            (compute_cost,
             max_n_succ_stages) = interpret_profile_result_training_2d(
                 profile_results, num_layers, num_submesh_choices,
                 num_autosharding_configs)
    elif auto_stage_option.layer_profile_mode == "individual":
        if inference_mode:
            compute_cost, _ = interpret_profile_result_inference_1d(
                profile_results, num_layers, num_submesh_choices,
                num_autosharding_configs)
            max_n_succ_stages = None
        else:
            (compute_cost,
             max_n_succ_stages) = interpret_profile_result_training_1d(
                 profile_results, num_layers, num_submesh_choices,
                 num_autosharding_configs)
    else:
        raise ValueError(f"Unknown layer profile mode: "
                         f"{auto_stage_option.layer_profile_mode}")

    return compute_cost, max_n_succ_stages


def select_module_layers(layers: Sequence[JaxPipelineComputation],
                         layer_indices: Sequence[int],
                         accumulator_mapping: Dict[Var, Var],
                         acc_grad_outvars: Sequence[Var]):
    """
    For each module, select the layers and get the accumulator mapping and
    required outvars for each module.

    Args:
        layers: all layers.
        layer_indices: a list of layer ids within the module.
        accumulator_mapping: the mapping from accumulator input to output,
            used to determine the donation.
        acc_grad_invars: the invars of the accumulator gradient layers.
        acc_grad_outvars: the outvars of the accumulator gradient layers.

    Returns:
        module: a list of layers that belong to the module.
        module_accumulator_mappings: accumulator mapping for the module.
        module_required_outvars: required outvars for the module.
    """
    reversed_accumulator_mapping = {
        v: k for k, v in accumulator_mapping.items()
    }

    gensym_fn = gensym([layer.closed_jaxpr().jaxpr for layer in layers])
    num_layers = len(layers)
    local_used = OrderedSet()
    new_layers = []
    module_required_outvars = OrderedSet()
    module_accumulator_mapping = {}
    used_by_other_layers_set = OrderedSet(acc_grad_outvars)
    for layer_id in reversed(range(num_layers)):
        layer = layers[layer_id]
        if layer_id not in layer_indices:
            used_by_other_layers_set.update(layer.invars)
            continue
        layer_donation, new_layer = (
            get_local_donation_mapping_and_add_missing_invars(
                layer, reversed_accumulator_mapping, gensym_fn))
        for invar in layer_donation:
            assert (invar not in local_used and
                    invar not in used_by_other_layers_set)

        required_outvars = [
            var for var in new_layer.outvars if var in used_by_other_layers_set
        ]
        module_accumulator_mapping.update(layer_donation)
        module_required_outvars.update(required_outvars)
        local_used.update(new_layer.invars)
        new_layers.append(new_layer)
    return (reversed(new_layers), module_accumulator_mapping,
            module_required_outvars)


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


def generate_stage_info(all_layers, selected_indices,
                        global_accumulator_mapping, acc_grad_invars,
                        acc_grad_outvars, name, apply_grad_layers,
                        apply_grad_info):
    """Combine selected layers together for profiling."""
    modules = []
    module_accumulator_mappings = []
    module_required_outvars = []
    for layer_indices in selected_indices:
        module, module_accumulator_mapping, required_outvars = (
            select_module_layers(all_layers, layer_indices,
                                 global_accumulator_mapping, acc_grad_outvars))
        modules.append(module)
        module_accumulator_mappings.append(module_accumulator_mapping)
        module_required_outvars.append(required_outvars)

    n_modules = len(modules)
    module_jaxprs = [
        [layer.closed_jaxpr() for layer in layers] for layers in modules
    ]

    module_names = [f"{name}_acc_grad_{i}" for i in range(n_modules)]
    module_merged_jaxprs = []
    module_profile_configs = []

    all_modules_donation_mapping = {}
    all_modules_donate_invars = []
    all_modules_outvars = OrderedSet()
    all_modules_acc_grad_outvars_indices = []
    acc_grad_invars_set = OrderedSet(acc_grad_invars)
    acc_grad_outvars_set = OrderedSet(acc_grad_outvars)
    for module_name, jaxprs, accumulator_mapping, required_outvars in zip(
            module_names, module_jaxprs, module_accumulator_mappings,
            module_required_outvars):
        merged_jaxpr = merge_marked_jaxprs_with_named_call(
            jaxprs, required_outvars, accumulator_mapping, module_name)
        outvars_set = set(merged_jaxpr.jaxpr.outvars)
        is_donated = tuple(invar in accumulator_mapping and
                           accumulator_mapping[invar] in outvars_set
                           for invar in merged_jaxpr.jaxpr.invars)
        acc_grad_invars_indices = tuple(
            i for i, outvar in enumerate(merged_jaxpr.jaxpr.invars)
            if outvar in acc_grad_invars_set)
        acc_grad_outvars_indices = tuple(
            i for i, outvar in enumerate(merged_jaxpr.jaxpr.outvars)
            if outvar in acc_grad_outvars_set)
        invar_names = tuple(repr(var) for var in merged_jaxpr.jaxpr.invars)
        outvar_names = tuple(repr(var) for var in merged_jaxpr.jaxpr.outvars)
        invar_avals = tuple(var.aval for var in merged_jaxpr.jaxpr.invars)
        outvar_avals = tuple(var.aval for var in merged_jaxpr.jaxpr.outvars)
        profile_config = ModuleProfileConfig(invar_names, outvar_names,
                                             invar_avals, outvar_avals,
                                             is_donated,
                                             acc_grad_invars_indices,
                                             acc_grad_outvars_indices)
        module_merged_jaxprs.append(merged_jaxpr)
        module_profile_configs.append(profile_config)
        all_modules_donate_invars.append(is_donated)
        all_modules_donation_mapping.update(accumulator_mapping)
        all_modules_outvars.update(merged_jaxpr.jaxpr.outvars)
        all_modules_acc_grad_outvars_indices.append(acc_grad_outvars_indices)

    if len(apply_grad_layers) > 0:
        apply_grad_donation, apply_grad_outvars = apply_grad_info
        apply_grad_module_name = "_".join([name, APPLY_GRAD_MARKER_SUFFIX])
        merged_apply = merge_marked_jaxprs_with_named_call(
            [layer.closed_jaxpr() for layer in apply_grad_layers],
            apply_grad_outvars, apply_grad_donation, name + "_apply")
        outvars_set = set(merged_apply.jaxpr.outvars)
        is_donated = tuple(invar in apply_grad_donation and
                           apply_grad_donation[invar] in outvars_set
                           for invar in merged_apply.jaxpr.invars)
        apply_only_invars = OrderedSet(merged_apply.jaxpr.invars)
        for module_jaxpr in module_merged_jaxprs:
            apply_only_invars = apply_only_invars.difference(
                module_jaxpr.jaxpr.invars)
            apply_only_invars = apply_only_invars.difference(
                module_jaxpr.jaxpr.outvars)
        apply_info = ApplyGradConfig(merged_apply.jaxpr.invars,
                                     apply_only_invars)
        module_names.append(apply_grad_module_name)
        module_merged_jaxprs.append(merged_apply)
        all_modules_donate_invars.append(is_donated)
        all_modules_donation_mapping.update(apply_grad_donation)
        all_modules_outvars.update(merged_apply.jaxpr.outvars)
    else:
        apply_info = None

    all_modules_merged_jaxpr, all_modules_is_donated = (
        merge_unmarked_with_call(module_merged_jaxprs, module_names,
                                 all_modules_outvars,
                                 all_modules_donation_mapping))
    hlo = jaxpr_to_hlo(name, all_modules_merged_jaxpr, all_modules_is_donated)
    compile_config = CompileConfig(hlo, module_names, all_modules_donate_invars,
                                   all_modules_acc_grad_outvars_indices)
    stage_config = StageConfig(n_modules, compile_config,
                               module_profile_configs, apply_info)
    return stage_config


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


def _get_sharded_sizes(sharding_specs, avals, logical_mesh_shape):
    """Compute bytes of avals with given sharding proto and logical
    mesh."""

    def get_byte(shape, dtype):
        return np.prod(shape) * np.dtype(dtype).itemsize

    if len(avals) == 0:
        return ()

    if np.prod(logical_mesh_shape) == 1:
        return tuple(get_byte(aval.shape, aval.dtype) for aval in avals)

    sharded_shapes = [
        get_shard_shape(aval, spec)
        for aval, spec in zip(avals, sharding_specs)
    ]

    return tuple(
        get_byte(shape, aval.dtype)
        for shape, aval in zip(sharded_shapes, avals))


def get_sharded_size_by_proto(serialized_proto,
                              avals,
                              logical_mesh_shape,
                              tuple_proto=True):
    """Compute bytes of serialized proto."""

    if len(avals) == 0:
        return ()

    if np.prod(logical_mesh_shape) == 1:
        sharding_specs = None
    else:
        if tuple_proto:
            hlo_sharding = xe.HloSharding(serialized_proto[0])
            sharding_specs = hlo_sharding_to_sharding_spec(
                hlo_sharding, avals, logical_mesh_shape)
        else:
            sharding_specs = [
                hlo_sharding_to_sharding_spec(xe.HloSharding(proto), aval,
                                              logical_mesh_shape)
                for (proto, aval) in zip(serialized_proto, avals)
            ]
    return _get_sharded_sizes(sharding_specs, avals, logical_mesh_shape)


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
    ordered_selected_avals = [v.aval for v in ordered_selected_vars]
    ordered_selected_names = [repr(v) for v in ordered_selected_vars]
    return (ordered_selected_names,
            _get_sharded_sizes(selected_sharding_specs, ordered_selected_avals,
                               logical_mesh_shape))
