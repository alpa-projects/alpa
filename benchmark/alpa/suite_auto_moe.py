"""Benchmark suites for moe with auto parallelization."""
from suite_manual_moe import moe_specs

max_global_batch_size = 1024
expert_group_size = 2048

search_global_config_dict = {
    "auto_stage_construction_imbalance_tolerance": 1.0,
    "logical_mesh_search_space": "all",
    "submesh_choices_mode": "small_power_of_two",
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}


prefer_reduce_scatter = True
use_remat = True


def get_search_cases(model_name, num_micro_batches_list, num_auto_layers_list):
    return [(max_global_batch_size, *moe_specs[model_name], expert_group_size,
             num_micro_batches, "search",
             (prefer_reduce_scatter, use_remat,
              num_auto_layers, search_global_config_dict))
            for num_micro_batches in num_micro_batches_list
            for num_auto_layers in num_auto_layers_list]


def get_solution_case(model_name, num_micro_batches, num_auto_layers,
                      forward_stage_layer_ids,
                      sub_physical_mesh_shapes, sub_logical_mesh_shapes,
                      submesh_autosharding_option_dicts):
    return [(max_global_batch_size, *moe_specs[model_name], expert_group_size,
             num_micro_batches, "load_solution",
             (prefer_reduce_scatter, use_remat, num_auto_layers,
              forward_stage_layer_ids,
              sub_physical_mesh_shapes, sub_logical_mesh_shapes,
              submesh_autosharding_option_dicts))]

# Temporary debug suite
tmp_suite = {
}


# Performance test with search solutions found for p3.16xlarge
perf_test_suite = {
1: get_solution_case("380M", 512,
    1, [[0]],
    [(1, 1)], [(1, 1)],
    [{}]),
2: get_solution_case("690M", 32,
    8, [[0, 1, 2, 3, 4, 5, 6, 7]],
    [(1, 2)], [(2, 1)],
    [{'force_batch_dim_to_mesh_dim': 0}]),
4: get_solution_case("1.3B", 32,
    8, [[0, 1, 2, 3], [4, 5, 6, 7]],
    [(1, 2)] * 2, [(2, 1)] * 2,
    [{'force_batch_dim_to_mesh_dim': 0}] * 2),
8: get_solution_case("2.4B", 32,
    8, [[0, 1, 2, 3], [4, 5, 6, 7]],
    [(1, 4)] * 2, [(4, 1)] * 2,
    [{'force_batch_dim_to_mesh_dim': 0}] * 2),
16: get_solution_case("10B", 16,
    8, [[0, 1, 2, 3], [4, 5, 6, 7]],
    [(1, 8)] * 2, [(8, 1)] * 2,
    [{}] * 2),
32: get_solution_case("27B", 128,
    8, [[0], [1], [2], [3], [4], [5], [6], [7]],
    [(1, 4)] * 8, [(4, 1)] * 8,
    [{}] * 8),
64: get_solution_case("70B", 64,
    8, [[0], [1], [2], [3], [4], [5], [6], [7]],
    [(1, 8)] * 8, [(8, 1)] * 8,
    [{}] * 8),
}

# Grid search on hyperparameters
grid_search_suite = {
2: (get_search_cases("690M", [16, 32, 64], [8])),
4: (get_search_cases("1.3B", [16, 32, 64], [8])),
8: (get_search_cases("2.4B", [16, 32, 64], [8])),
16: (get_search_cases("10B", [16, 32, 64], [8])),
32: (get_search_cases("27B", [32, 64, 128], [4, 8, 16])),
64: (get_search_cases("70B", [64], [8, 16, 32])),   # submesh_choices_mode: "small_power_of_two", max num_cpus = 20
}
