"""Suites for wresnet benchmarking."""
from collections import namedtuple
import numpy as np

from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs)

UNetModelConfig = namedtuple(
    "UNetModelConfig",
    ["image_size", "channel_size", "block_cnt", "dtype", "num_layers"])

# block cnt->manual layers: {4: 13, }
unet_specs = {
    # #Params: sample size, first channel's size, block cnt, dtype
    "470M": UNetModelConfig(32, 320, 4, np.float32, 13),
    "1B": UNetModelConfig(32, 480, 4, np.float32, 13),
    "1.2B": UNetModelConfig(32, 512, 4, np.float32, 13),
    "1.8B": UNetModelConfig(32, 640, 4, np.float32, 13),
    "2B": UNetModelConfig(32, 672, 4, np.float32, 13),
}

prefer_reduce_scatter = False
use_remat = True
force_batch_dim_mapping = False

auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "single_node_model_parallel",
    "stage_imbalance_tolerance": 0.25,
    "use_hlo_cost_model": False,
    "profiling_database_filename": None,
}


def get_num_auto_layers(name):
    return int(unet_specs[name].block_cnt * 1.5)


def get_search_cases(model_name, max_global_batch_size, num_micro_batches_list):
    num_auto_layers = get_num_auto_layers(model_name)
    return [
        BenchmarkCase(
            max_global_batch_size, unet_specs[model_name], num_micro_batches,
            "search",
            SearchParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option))
        for num_micro_batches in num_micro_batches_list
    ]


def get_solution_case(model_name, max_global_batch_size, num_micro_batches,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts):
    num_auto_layers = get_num_auto_layers(model_name)
    return [
        BenchmarkCase(
            max_global_batch_size, unet_specs[model_name], num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts))
    ]


# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor,
# NB = num_micro_batches, PM = parallel_mode
# L_Shape = logical_mesh_shape
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,

force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}

# Performance test with shard parallel
tmp_suite = {}

# Performance test with shard parallel
# key = the number of gpus, value = a list of cases
# B,    I,   L,   C,   W, dtype,  NB, PM,          RS,    Remat, L_shape, FM
perf_test_2d_suite = {}

# Performance test with search solutions found for p3.16xlarge
perf_test_auto_suite = {
    2:
        get_solution_case("470M", 256, 4,
                          [list(range(7)), list(range(7, 13))], [(1, 1)] * 2,
                          [(1, 1)] * 2, [{}] * 2),
    4:
        get_solution_case("1B", 2048, 32,
                          [list(range(8)), list(range(8, 13))], [(1, 2)] * 2,
                          [(1, 2)] * 2, [{}] * 2),
    8:
        get_solution_case("2B", 2048, 32,
                          [list(range(9)), list(range(9, 13))], [(1, 4)] * 2,
                          [(1, 4)] * 2, [{}] * 2),
}

# Grid search on hyperparameters
# key = the number of gpus, value = a list of cases
# model_name, B, NB
grid_search_auto_suite = {
    4: get_search_cases("1B", 256, [
        16,
    ])
}
