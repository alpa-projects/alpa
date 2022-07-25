"""Options of a benchmark case."""
from typing import Optional
from collections import namedtuple

import numpy as np

from alpa import (AutoShardingOption, PipeshardParallel, ManualStageOption,
                  AutoStageOption, AutoLayerOption)
from alpa.timer import timers
from alpa.util import print_used_time, to_str_round

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


def get_parallel_method(benchmark_case: BenchmarkCase,
                        num_devices_per_host: Optional[int] = None,
                        allow_mixed_mesh_shape: bool = False,
                        use_fine_grained_remat: bool = False):
    """Create the parallel method of a benchmark case.

    Args:
        benchmark_case: The benchmark case.
        num_devices_per_host: The number of devices per host, used in uniform
          parallel mode.
        allow_mixed_mesh_shape: Whether to allow the mixed mesh shape in
          the autosharding pass.
        use_fine_grained_remat: Whether to use fine grained remat. If True,
          the remat pass in auto layer pass will be skipped. This option only
          works for load_solution parallel mode now.
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
        auto_stage_option["cached_compute_cost"] = None
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            layer_option=AutoLayerOption(layer_num=num_auto_layers,
                                         remat_layer=use_remat),
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
        if use_fine_grained_remat:
            use_remat = False
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            layer_option=AutoLayerOption(layer_num=num_auto_layers,
                                         remat_layer=use_remat),
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


def compile_and_benchmark_executable(parallel_mode, niter, train_step, state,
                                     other_train_step_inputs):
    print_used_time(None)

    executable = train_step.get_executable(state, *other_train_step_inputs)
    print_used_time("Compile (driver)")

    if parallel_mode == "search":
        compilation_times = {
            k: timers(k).elapsed() for k in [
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

    # Benchmark latency without driver overhead
    for i in range(niter):
        print(f"Iteration {i} ...")
        state = train_step(state, *other_train_step_inputs)
        executable.sync()

    latencies = executable.get_execution_time_costs()[1:]
    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()

    # Benchmark latency with driver overhead
    if False:
        global_config.use_dummy_value_for_benchmarking = False
        global_config.pipeline_sync_for_timer = False
        number = niter
        executable.sync()
        tic = time.time()
        for i in range(number):
            state = train_step(state, batch, rngkey)
        executable.sync()
        e2e_latency = (time.time() - tic) / number
        print(f"latency with dirver overhead: {e2e_latency:.3f}")
    print_used_time("Benchmark")

    return latencies, max_mem_allocated, compilation_times, executable
