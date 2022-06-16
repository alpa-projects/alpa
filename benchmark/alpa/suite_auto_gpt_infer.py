"""Benchmark suites for gpt with auto parallelization."""
from suite_manual_gpt import gpt_specs

max_global_batch_size = 1024

auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "all",
    "auto_stage_imbalance_tolerance": 1.0,
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}


prefer_reduce_scatter = True
use_remat = True

def get_solution_case(model_name, num_micro_batches, num_auto_layers,
                      forward_stage_layer_ids,
                      submesh_physical_shapes, submesh_logical_shapes,
                      submesh_autosharding_option_dicts):
    return [(max_global_batch_size, *gpt_specs[model_name],
             num_micro_batches, "load_solution",
             (prefer_reduce_scatter, use_remat, num_auto_layers,
              (forward_stage_layer_ids,
               submesh_physical_shapes, submesh_logical_shapes,
               submesh_autosharding_option_dicts)))]


# Temporary debug suite
tmp_suite = {
}


# Performance test with search solutions found for p3.16xlarge
perf_test_suite = {
# 1: get_solution_case("350M", 512,
#                      1, [[0]],
#                      [(1, 1)], [(1, 1)],
#                      [{}]),
2: get_solution_case("760M", 128,
                     2, [[0], [1]],
                     [(1, 1)] * 2, [(1, 1)] * 2,
                     [{'force_batch_dim_to_mesh_dim': 0}] * 2),
# 4: get_solution_case("1.3B", 128,
#                      6, [[0, 1, 2], [3, 4, 5]],
#                      [(1, 2)] * 2, [(2, 1)] * 2,
#                      [{'force_batch_dim_to_mesh_dim': 0}] * 2),
# 8: get_solution_case("2.6B", 128,
#                      8, [[0, 1], [2, 3], [4, 5, 6, 7]],
#                      [(1, 2), (1, 2), (1, 4)], [(2, 1), (2, 1), (4, 1)],
#                      [{'force_batch_dim_to_mesh_dim': 0}, {}, {}]),
# 16: get_solution_case("6.7B", 64,
#                       8, [[0, 1, 2, 3], [4, 5, 6, 7]],
#                       [(1, 8)] * 2, [(2, 4)] * 2,
#                       [{'force_batch_dim_to_mesh_dim': 0}] * 2),
# 32: get_solution_case("15B", 128,
#                       16, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
#                       [(1, 8)] * 4, [(2, 4)] * 4,
#                       [{'force_batch_dim_to_mesh_dim': 0}] * 4),
# 64: get_solution_case("39B", 1024,
#                       16, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
#                       [(1, 4)] * 16, [(1, 4)] * 16,
#                       [{'force_batch_dim_to_mesh_dim': 0}] * 16),
}