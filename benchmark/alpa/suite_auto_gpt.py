"""Benchmark suites for gpt with auto parallelization."""
from suite_manual_gpt import gpt_specs

dummy_arguments = (0, 0, 0, 0)  # LD0, LD1, PD0, PD1, not used for auto
fixed_params = (False, True, True) # FM, Remat, RS
max_global_batch_size = 1024


default_overwrite_dict = {
    "auto_stage_construction_imbalance_tolerance": 1.0,
    "logical_mesh_search_space": "all",
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
    "submesh_choices_mode": "small_power_of_two",
}


def get_benchmark_cases(model_name, n_microbatches, num_layers,
                       pipeline_stage_mode="auto_gpipe",
                       overwrite_global_config_dict=None):
    if overwrite_global_config_dict is None:
        overwrite_global_config_dict = default_overwrite_dict
    return [(max_global_batch_size, *gpt_specs[model_name],
             *dummy_arguments, num_layer, n_microbatch, *fixed_params,
             pipeline_stage_mode, overwrite_global_config_dict)
            for n_microbatch in n_microbatches
            for num_layer in num_layers]


# Temporary debug suite
tmp_suite = {
1: get_benchmark_cases("125M", [64], [6]),
2: get_benchmark_cases("350M", [64], [6]),
4: get_benchmark_cases("760M", [64], [6], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2], [3, 4, 5]],
    "sub_physical_mesh_shapes": [(1, 2)] * 2,
    "sub_logical_mesh_shapes": [(2, 1)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2,
}),
8: get_benchmark_cases("2.6B", [128], [8]),
16: get_benchmark_cases("6.7B", [256], [8]),
32: get_benchmark_cases("15B", [256], [16], "manual_gpipe", {
    "forward_stage_layer_ids": [[2 * i, 2 * i + 1] for i in range(8)],
    "sub_physical_mesh_shapes": [(1, 4)] * 8,
    "sub_logical_mesh_shapes": [(4, 1), (4, 1), (4, 1), (4, 1), (1, 4), (4, 1), (1, 4), (1, 4)],
    "submesh_autosharding_option_dicts": [{}, {}, {}, {}, {'force_batch_dim_to_mesh_dim': 0}, {},
                                          {'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}]}),
64: get_benchmark_cases("39B", [256], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[i] for i in range(8)],
    "sub_physical_mesh_shapes": [(1, 8)] * 8,
    "sub_logical_mesh_shapes": [(1, 8)] * 8,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 8,
})
}

# Performance test with search solutions found for p3.16xlarge
perf_test_suite = {
1: get_benchmark_cases("350M", [512], [1], "manual_gpipe", {
    "forward_stage_layer_ids": [[0]],
    "sub_physical_mesh_shapes": [(1, 1)],
    "sub_logical_mesh_shapes": [(1, 1)],
    "submesh_autosharding_option_dicts": [{}],
}),
2: get_benchmark_cases("760M", [128], [6], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2], [3, 4, 5]],
    "sub_physical_mesh_shapes": [(1, 1)] * 2,
    "sub_logical_mesh_shapes": [(1, 1)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2,
}),
4: get_benchmark_cases("1.3B", [128], [6], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2], [3, 4, 5]],
    "sub_physical_mesh_shapes": [(1, 2)] * 2,
    "sub_logical_mesh_shapes": [(2, 1)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2,
}),
8: get_benchmark_cases("2.6B", [128], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1], [2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 2), (1, 2), (1, 4)],
    "sub_logical_mesh_shapes": [(2, 1), (2, 1), (4, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {}, {}],
}),
16: get_benchmark_cases("6.7B", [64], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 8)] * 2,
    "sub_logical_mesh_shapes": [(2, 4)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2,
}),
32: get_benchmark_cases("15B", [128], [16], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 8)] * 4,
    "sub_logical_mesh_shapes": [(2, 4)] * 4,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 4,
}),
64: get_benchmark_cases("39B", [1024], [16], "manual_gpipe", {
    "forward_stage_layer_ids": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
    "sub_physical_mesh_shapes": [(1, 4)] * 16,
    "sub_logical_mesh_shapes": [(1, 4)] * 16,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 16,
}),
}


# Grid search on hyperparameters
grid_search_suite = {
2: (get_benchmark_cases("760M", [32, 64, 128, 256], [6]) +
    get_benchmark_cases("760M", [32, 64], [12])),
4: (get_benchmark_cases("1.3B", [32, 64, 128], [6]) +
    get_benchmark_cases("1.3B", [32, 64], [12])),
8: (get_benchmark_cases("2.6B", [64, 128, 256], [8]) +
    get_benchmark_cases("2.6B", [64, 128], [16])),
16: get_benchmark_cases("6.7B", [32, 64, 128, 256], [8]),
32: get_benchmark_cases("15B", [64, 128, 256, 512], [16]),
64: get_benchmark_cases("39B", [128, 256, 512, 1024], [8]),
}


# Small test cases for correctness test
correctness_test_suite = {
8: get_benchmark_cases("2.6B", [128], [8]),
}
