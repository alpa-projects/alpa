"""Options of a benchmark case."""
from collections import namedtuple
import json
import os
import time
from typing import Optional, Dict, Any, List

import numpy as np
import jax
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten

import alpa
from alpa import (AutoShardingOption, ShardParallel, PipeshardParallel,
                  ManualStageOption, AutoStageOption, AutoLayerOption,
                  global_config, PhysicalDeviceMesh)
from alpa.timer import timers
from alpa.util import (print_used_time, to_str_round,
                       count_communication_primitives, GB)

BenchmarkCase = namedtuple("BenchmarkCase", [
    "batch_size", "model_config", "num_micro_batches", "parallel_mode",
    "parallel_args"
])

ShardParallelArgs = namedtuple("ShardParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "logical_mesh_shape",
    "force_batch_dim_mapping"
])

UniformParallelArgs = namedtuple("UniformParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "dp", "op", "pp",
    "force_batch_dim_mapping"
])

SearchParallelArgs = namedtuple("SearchParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers", "auto_stage_option"
])

LoadSolutionParallelArgs = namedtuple("LoadSolutionParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers",
    "forward_stage_layer_ids", "submesh_physical_shapes",
    "submesh_logical_shapes", "submesh_autosharding_option_dicts"
])


def get_pipeshard_parallel_method(benchmark_case: BenchmarkCase,
                                  num_devices_per_host: Optional[int] = None,
                                  allow_mixed_mesh_shape: bool = False,
                                  use_fine_grained_remat: bool = False,
                                  pipeline_schedule: str = "1f1b"):
    """Create the parallel method of a benchmark case.

    Args:
        benchmark_case: The benchmark case.
        num_devices_per_host: The number of devices per host, used in uniform
          parallel mode.
        allow_mixed_mesh_shape: Whether to allow the mixed mesh shape in
          the autosharding pass.
    """

    num_micro_batches = benchmark_case.num_micro_batches
    parallel_mode = benchmark_case.parallel_mode
    parallel_args = benchmark_case.parallel_args

    if parallel_mode == "search":
        assert isinstance(parallel_args, SearchParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
         auto_stage_option) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        remat_mode = "coarse_grained_remat" if use_remat else "none"
        auto_stage_option["cached_profile_result"] = None
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(layer_num=num_auto_layers,
                                         remat_mode=remat_mode),
            stage_option=AutoStageOption(**auto_stage_option))
    elif parallel_mode == "load_solution":
        assert isinstance(parallel_args, LoadSolutionParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
         forward_stage_layer_ids, submesh_physical_shapes,
         submesh_logical_shapes,
         submesh_autosharding_option_dicts) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        if use_remat:
            remat_mode = ("fine_grained_remat"
                          if use_fine_grained_remat else "coarse_grained_remat")
        else:
            remat_mode = "none"
        model_num_layers = benchmark_case.model_config.num_layers
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(
                layer_num=num_auto_layers,
                remat_mode=remat_mode,
                fine_grained_remat_layer_num=model_num_layers),
            stage_option=ManualStageOption(forward_stage_layer_ids,
                                           submesh_physical_shapes,
                                           submesh_logical_shapes,
                                           submesh_autosharding_option_dicts))
    elif parallel_mode == "uniform":
        assert isinstance(parallel_args, UniformParallelArgs)
        (prefer_reduce_scatter, use_remat, dp, op, pp,
         force_batch_dim_mapping) = parallel_args
        as_option = AutoShardingOption(
            prefer_reduce_scatter=prefer_reduce_scatter,
            allow_mixed_mesh_shape=allow_mixed_mesh_shape,
        )
        if force_batch_dim_mapping:
            as_option.force_batch_dim_to_mesh_dim = 0
        add_manual_layer_marker = True
        add_manual_remat = use_remat

        logical_mesh_shape = (dp, op)
        num_manual_pipeline_stages = pp
        num_mesh_devices = np.prod(logical_mesh_shape)
        assert num_devices_per_host is not None
        if num_mesh_devices <= num_devices_per_host:
            physical_mesh_shape = (1, num_mesh_devices)
        else:
            assert num_mesh_devices % num_devices_per_host == 0
            physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
                                   num_devices_per_host)

        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=as_option,
            pipeline_schedule=pipeline_schedule,
            layer_option="manual",
            stage_option=ManualStageOption(
                forward_stage_layer_ids=[[i] for i in range(pp)],
                submesh_physical_shapes=[physical_mesh_shape] * pp,
                submesh_logical_shapes=[logical_mesh_shape] * pp,
                submesh_autosharding_option_dicts=[{}] * pp))
    else:
        raise ValueError(f"Invalid parallel mode: {parallel_mode}")

    return (method, add_manual_remat, add_manual_layer_marker,
            num_manual_pipeline_stages)


def get_shard_parallel_method(benchmark_case: BenchmarkCase,
                              physical_mesh: PhysicalDeviceMesh,
                              logical_mesh_options: Dict[str, Any] = None):
    """Create the parallel method of a benchmark case.

    Args:
        benchmark_case: The benchmark case.
        num_devices_per_host: The number of devices per host, used in uniform
          parallel mode.
        allow_mixed_mesh_shape: Whether to allow the mixed mesh shape in
          the autosharding pass.
    """
    print_used_time(None)

    num_micro_batches = benchmark_case.num_micro_batches
    parallel_mode = benchmark_case.parallel_mode
    parallel_args = benchmark_case.parallel_args

    if isinstance(parallel_args, ShardParallelArgs):
        (prefer_reduce_scatter, use_remat, logical_mesh_shape,
         force_batch_dim_mapping) = parallel_args
    elif isinstance(parallel_args, UniformParallelArgs):
        (prefer_reduce_scatter, use_remat, dp, op, pp,
         force_batch_dim_mapping) = parallel_args
        assert pp == 1, "Do not support pipeline parallelism for shard parallel"
        logical_mesh_shape = (dp, op)
    else:
        raise ValueError(f"Unsupported parallel mode: {parallel_mode}")

    # Parallel configs
    if num_micro_batches > 1:
        grad_func = alpa.grad
    else:
        num_micro_batches = None
        grad_func = jax.grad

    as_option = AutoShardingOption()
    if force_batch_dim_mapping:  # Always map batch dim to mesh dim 0
        as_option.force_batch_dim_to_mesh_dim = 0
    as_option.prefer_reduce_scatter = prefer_reduce_scatter
    if parallel_mode == "zero-3":
        as_option.force_zero_stage_3 = True
    elif parallel_mode in ["shard-largest"]:
        as_option.force_simple_heuristic = "largest"

    if logical_mesh_options is None:
        logical_mesh_options = {}
    logical_mesh = physical_mesh.get_logical_mesh(logical_mesh_shape,
                                                  **logical_mesh_options)
    method = ShardParallel(devices=logical_mesh,
                           num_micro_batches=num_micro_batches,
                           auto_sharding_option=as_option)
    print_used_time("Setup device mesh")

    return method, grad_func


def benchmark_training_executable(niter,
                                  train_step,
                                  executable,
                                  state,
                                  other_train_step_inputs,
                                  profile_driver_time=False):
    print_used_time(None)

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if profile_driver_time:
        # Benchmark latency with driver overhead
        global_config.use_dummy_value_for_benchmarking = False
        global_config.shard_parallel_sync_for_timer = False
        print("Warmup")
        for i in range(warmup):
            state = train_step(state, *other_train_step_inputs)
        executable.sync()
        niter -= warmup
        print("Benchmark")
        tic = time.time()
        for i in range(niter):
            state = train_step(state, *other_train_step_inputs)
        executable.sync()
        e2e_latency = (time.time() - tic) / niter
        latencies = [e2e_latency]
        print(f"latency with driver overhead: {e2e_latency:.3f}")
    else:
        # Benchmark latency without driver overhead
        for i in range(niter):
            print(f"Iteration {i} ...")
            state = train_step(state, *other_train_step_inputs)
            if isinstance(state, tuple):
                # In case the train_step returns extra info (e.g. loss),
                # Get the actual state out.
                state = state[0]
            executable.sync()

        latencies = executable.get_execution_time_costs()[warmup:]

    print_used_time("Benchmark")

    return latencies


def benchmark_inference_executable(niter,
                                   infer_step,
                                   executable,
                                   params,
                                   other_infer_step_inputs,
                                   profile_driver_time=False):
    print_used_time(None)

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if profile_driver_time:
        # Benchmark latency with streaming
        for i in range(warmup):
            _ = infer_step(params, *other_infer_step_inputs)
        executable.sync()
        niter -= warmup

        # Benchmark latency
        losses = []
        start_time = time.time()
        latencies = []
        for i in range(niter):
            print(f"Iteration {i} ...")
            loss = infer_step(params, *other_infer_step_inputs)
            loss.prefetch()
            losses.append(loss)
        for i, loss in enumerate(losses):
            _ = loss._value
            end_time = time.time()
            latencies.append(end_time - start_time)
            start_time = end_time
    else:
        for i in range(niter):
            print(f"Iteration {i} ...")
            _ = infer_step(params, *other_infer_step_inputs)
            executable.sync()

        latencies = executable.get_execution_time_costs()[warmup:]

    print_used_time("Benchmark")

    return latencies


def compile_pipeshard_executable(parallel_mode, train_step, state,
                                 other_train_step_inputs):
    print_used_time(None)

    executable = train_step.get_executable(state, *other_train_step_inputs)
    print_used_time("Compile (driver)")

    if parallel_mode == "search":
        compilation_times = {
            k: timers(k).elapsed(mode="sum") for k in [
                "stage-construction", "stage-construction-dp",
                "stage-construction-compilation", "stage-construction-profiling"
            ]
        }
        print(
            f"compilation time breakdown: {to_str_round(compilation_times, 2)}")
    else:
        compilation_times = None

    executable.dump_debug_info("tmp")
    executable.sync()
    print_used_time("Compile (worker)")
    return executable, compilation_times


def compile_shard_executable(physical_mesh, train_step, state,
                             other_train_step_inputs):
    print_used_time(None)
    executable = train_step.get_executable(state, *other_train_step_inputs)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Check sharding strategy
    alloc_mem = executable.get_total_allocation_size()
    ilp_objective = executable.auto_sharding_objective or 0.0
    executable.dump_debug_info("tmp")
    hlo_text = executable.get_hlo_text()
    (n_total, n_all_reduce, n_all_gather, n_reduce_scatter,
     n_all_to_all) = count_communication_primitives(hlo_text)

    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")
    print(f"alloc_mem: {alloc_mem / GB:.2f} GB")
    return executable, ilp_objective, alloc_mem


def compile_and_benchmark_pipeshard_training_executable(
        parallel_mode,
        niter,
        train_step,
        state,
        other_train_step_inputs,
        profile_driver_time=False):
    executable, compilation_times = compile_pipeshard_executable(
        parallel_mode, train_step, state, other_train_step_inputs)
    latencies = benchmark_training_executable(
        niter,
        train_step,
        executable,
        state,
        other_train_step_inputs,
        profile_driver_time=profile_driver_time)
    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()

    return latencies, max_mem_allocated, compilation_times, executable


def compile_and_benchmark_shard_training_executable(physical_mesh,
                                                    niter,
                                                    train_step,
                                                    state,
                                                    other_train_step_inputs,
                                                    profile_driver_time=False):
    executable, ilp_objective, alloc_mem = compile_shard_executable(
        physical_mesh, train_step, state, other_train_step_inputs)
    latencies = benchmark_training_executable(
        niter,
        train_step,
        executable,
        state,
        other_train_step_inputs,
        profile_driver_time=profile_driver_time)
    peak_mem = max(physical_mesh.get_max_memory_allocated(), alloc_mem)
    return latencies, ilp_objective, peak_mem, executable


def compile_and_benchmark_pipeshard_inference_executable(
        parallel_mode,
        niter,
        infer_step,
        params,
        other_inference_step_inputs,
        profile_driver_time=False):
    executable, compilation_times = compile_pipeshard_executable(
        parallel_mode, infer_step, params, other_inference_step_inputs)

    # Preshard params
    executable.mesh_group.reset_memory_stats()
    params_ps = executable.get_input_placement_specs()[0]
    flat_params, in_tree = tree_flatten(params)
    flat_ps = tree_leaves(params_ps)
    params = tree_unflatten(
        in_tree,
        executable.mesh_group.shard_args_to_arrays(flat_ps, flat_params))
    print_used_time("Preshard (driver)")
    per_stage_weight_mem = executable.mesh_group.get_max_memory_allocated_per_mesh(
    )

    latencies = benchmark_inference_executable(
        niter,
        infer_step,
        executable,
        params,
        other_inference_step_inputs,
        profile_driver_time=profile_driver_time)
    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()
    per_stage_peak_mem = executable.mesh_group.get_max_memory_allocated_per_mesh(
    )

    return latencies, max_mem_allocated, compilation_times, executable, per_stage_weight_mem, per_stage_peak_mem


def compute_avg_stage_latencies(timelines: List[tuple]):
    stage_latencies = []
    for request_timeline in timelines:
        sorted_timeline = sorted(request_timeline, key=lambda x: x[0])
        stage_borders = [sorted_timeline[0][0]]
        for _, e, _, _ in sorted_timeline:
            stage_borders.append(e)
        stage_latency = [
            stage_borders[i + 1] - stage_borders[i]
            for i in range(len(stage_borders) - 1)
        ]
        stage_latencies.append(stage_latency)
    return np.mean(stage_latencies, axis=0)
