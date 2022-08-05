"""Benchmark suites for gpt with auto parallelization."""
from suite_manual_gpt import gpt_specs
from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs)

max_global_batch_size = 1024

auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "all",
    "stage_imbalance_tolerance": 1.0,
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}

prefer_reduce_scatter = True
use_remat = True


def get_search_cases(model_spec, num_micro_batches_list, num_auto_layers_list):
    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches, "search",
            SearchParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option))
        for num_micro_batches in num_micro_batches_list
        for num_auto_layers in num_auto_layers_list
    ]


def get_solution_case(model_spec, num_micro_batches, num_auto_layers,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts):
    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts))
    ]


force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}

# Temporary debug suite
tmp_suite = {}

# Performance test with search solutions found for p3.16xlarge
perf_test_suite = {
    1:
        get_solution_case(gpt_specs["350M"], 512, 1, [[0]], [(1, 1)], [(1, 1)],
                          [{}]),
    2:
        get_solution_case(gpt_specs["760M"], 128, 6, [[0, 1, 2], [3, 4, 5]],
                          [(1, 1)] * 2, [(1, 1)] * 2, [force_dp_dict] * 2),
    4:
        get_solution_case(gpt_specs["1.3B"], 128, 6, [[0, 1, 2], [3, 4, 5]],
                          [(1, 2)] * 2, [(2, 1)] * 2, [force_dp_dict] * 2),
    8:
        get_solution_case(gpt_specs["2.6B"], 128,
                          8, [[0, 1], [2, 3], [4, 5, 6, 7]], [(1, 2), (1, 2),
                                                              (1, 4)], [(2, 1),
                                                                        (2, 1),
                                                                        (4, 1)],
                          [force_dp_dict, {}, {}]),
    16:
        get_solution_case(gpt_specs["6.7B"], 64, 8,
                          [[0, 1, 2, 3], [4, 5, 6, 7]], [(1, 8)] * 2,
                          [(2, 4)] * 2, [force_dp_dict] * 2),
    32:
        get_solution_case(
            gpt_specs["15B"], 128, 16,
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [(1, 8)] * 4, [(2, 4)] * 4, [force_dp_dict] * 4),
    64:
        get_solution_case(gpt_specs["39B"], 1024,
                          16, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
                               [10], [11], [12], [13], [14], [15]],
                          [(1, 4)] * 16, [(1, 4)] * 16, [force_dp_dict] * 16),
}

# Grid search on hyperparameters
grid_search_suite = {
    2: (get_search_cases(gpt_specs["760M"], [32, 64, 128, 256], [6]) +
        get_search_cases(gpt_specs["760M"], [32, 64], [12])),
    4: (get_search_cases(gpt_specs["1.3B"], [32, 64, 128], [6]) +
        get_search_cases(gpt_specs["1.3B"], [32, 64], [12])),
    8: (get_search_cases(gpt_specs["2.6B"], [64, 128, 256], [8]) +
        get_search_cases(gpt_specs["2.6B"], [64, 128], [16])),
    16: get_search_cases(gpt_specs["6.7B"], [32, 64, 128, 256], [8]),
    32: get_search_cases(gpt_specs["15B"], [64, 128, 256, 512], [16]),
    64: get_search_cases(gpt_specs["39B"], [128, 256, 512, 1024], [8]),
}

# Small test cases for correctness test
correctness_test_suite = {
    8: get_search_cases(gpt_specs["2.6B"], [128], [8]),
}
