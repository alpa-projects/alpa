"""Benchmark suites for moe with auto parallelization."""
from suite_manual_moe import moe_specs

_ = None

dummy_arguments = (1, 1, 0, 0) # LD0, LD1, PD0, PD1, not used for auto
fixed_params = (False, True, True) # FM, Remat, RS
max_global_batch_size = 1024
expert_group_size = 2048


default_overwrite_dict = {
    "auto_stage_construction_imbalance_tolerance": 1.0,
    "logical_mesh_search_space": "all",
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}


def get_benchmark_cases(model_name, n_microbatches, num_layers,
                       pipeline_stage_mode="auto_gpipe",
                       overwrite_global_config_dict=None):
    if overwrite_global_config_dict is None:
        overwrite_global_config_dict = default_overwrite_dict
    return [(max_global_batch_size, *moe_specs[model_name], expert_group_size,
             *dummy_arguments, num_layer, n_microbatch, *fixed_params,
             pipeline_stage_mode, overwrite_global_config_dict)
            for n_microbatch in n_microbatches
            for num_layer in num_layers]


# Temporary debug suite
tmp_suite = {
1:  get_benchmark_cases("380M", [64], [4]),
2:  get_benchmark_cases("690M", [64], [4]),
4:  get_benchmark_cases("1.3B", [64], [8]),
8:  get_benchmark_cases("2.4B", [16], [8]),
16: get_benchmark_cases("10B",  [32], [8]),
32: get_benchmark_cases("27B",  [32], [4]),
}


# Performance test with search solutions found for p3.16xlarge
perf_test_suite = {
1: get_benchmark_cases("380M", [512], [1], "manual_gpipe", {
    "forward_stage_layer_ids": [[0]],
    "sub_physical_mesh_shapes": [(1, 1)],
    "sub_logical_mesh_shapes": [(1, 1)],
    "submesh_autosharding_option_dicts": [{}],
}),
2: get_benchmark_cases("690M", [32], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 2)],
    "sub_logical_mesh_shapes": [(2, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}],
}),
4: get_benchmark_cases("1.3B", [32], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 2)] * 2,
    "sub_logical_mesh_shapes": [(2, 1)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2,
}),
8: get_benchmark_cases("2.4B", [32], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 4)] * 2,
    "sub_logical_mesh_shapes": [(4, 1)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2,
}),
16: get_benchmark_cases("10B", [16], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 8)] * 2,
    "sub_logical_mesh_shapes": [(8, 1)] * 2,
    "submesh_autosharding_option_dicts": [{}] * 2,
}),
32: get_benchmark_cases("27B", [128], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0], [1], [2], [3], [4], [5], [6], [7]],
    "sub_physical_mesh_shapes": [(1, 4)] * 8,
    "sub_logical_mesh_shapes": [(4, 1)] * 8,
    "submesh_autosharding_option_dicts": [{}] * 8,
}),
}


# Grid search on hyperparameters
grid_search_suite = {
2: (get_benchmark_cases("690M", [16, 32, 64], [8])),
4: (get_benchmark_cases("1.3B", [16, 32, 64], [8])),
8: (get_benchmark_cases("2.4B", [16, 32, 64], [8])),
16: (get_benchmark_cases("10B", [16, 32, 64], [8])),
32: (get_benchmark_cases("27B", [32, 64, 128], [4, 8, 16])),
64: (get_benchmark_cases("70B", [64], [8, 16, 32])),   # submesh_choices_mode: "small_power_of_two", max num_cpus = 20
}
