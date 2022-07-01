"""Suites for wresnet benchmarking."""
_ = None

# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor,
# NB = num_micro_batches, PM = parallel_mode
# L_Shape = logical_mesh_shape
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,

wresnet_specs = {
    #    I,   L,   C,   W,  dtype,
    "250M": (224, 50, 160, 2, "fp32"),
    "500M": (224, 50, 224, 2, "fp32"),
    "1B": (224, 50, 320, 2, "fp32"),
    "2B": (224, 50, 448, 2, "fp32"),
    "4B": (224, 50, 640, 2, "fp32"),
    "6.8B": (224, 50, 320, 16, "fp32"),
    "13B": (224, 101, 320, 16, "fp32"),
}

prefer_reduce_scatter = True
use_remat = True

auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "single_node_model_parallel",
    "stage_imbalance_tolerance": 0.25,
    "use_hlo_cost_model": False,
    "profiling_database_filename": None,
}


def get_search_cases(model_name, max_global_batch_size, num_micro_batches_list):
    return [(max_global_batch_size, *wresnet_specs[model_name],
             num_micro_batches, "search", (prefer_reduce_scatter, use_remat,
                                           auto_stage_option))
            for num_micro_batches in num_micro_batches_list]


def get_solution_case(model_name, max_global_batch_size, num_micro_batches,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts):
    return [(max_global_batch_size, *wresnet_specs[model_name],
             num_micro_batches, "load_solution",
             (prefer_reduce_scatter, use_remat,
              (forward_stage_layer_ids, submesh_physical_shapes,
               submesh_logical_shapes, submesh_autosharding_option_dicts)))]


# Performance test with shard parallel
tmp_suite = {  # key = the number of gpus, value = a list of cases
}

# Performance test with shard parallel
perf_test_2d_suite = { # key = the number of gpus, value = a list of cases
    1: [
    #B,    I,   L,   C,   W, dtype,  NB, PM,          RS,    Remat, L_shape, FM
    (32,   224, 50,  160, 2, "fp32", 1,  "2d_shard", (False, False, (1, 1),  False)),
    (1536, 224, 50,  160, 2, "fp32", 48, "2d_shard", (False, False, (1, 1),  False)),
    ],

    4 : [
    (32,   224, 50,  320, 2, "fp32", 1,  "2d_shard", (False, False, (4, 1),  False)),
    (1536, 224, 50,  320, 2, "fp32", 48, "2d_shard", (False, False, (4, 1),  False)),
    (64,   224, 50,  320, 2, "fp32", 1,  "2d_shard", (False, False, (4, 1),  False)),
    (1536, 224, 50,  320, 2, "fp32", 24, "2d_shard", (False, False, (4, 1),  False)),
    ],

    8: [
    (64,   224, 50,  320, 2, "fp32", 1,  "2d_shard", (False, False, (8, 1),  False)),
    ],
}

# Performance test with search solutions found for p3.16xlarge
perf_test_auto_suite = {
    1:
        get_solution_case("250M", 1536, 24, [list(range(16))], [(1, 1)],
                          [(1, 1)], [{}]),
    2:
        get_solution_case("500M", 1536, 24, [list(range(16))], [(1, 2)],
                          [(1, 2)], [{}]),
    4:
        get_solution_case("1B", 1536, 24, [list(range(16))], [(1, 4)], [(1, 4)],
                          [{}]),
    8:
        get_solution_case(
            "2B", 1536, 24,
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
            [(1, 4), (1, 4)], [(4, 1), (1, 4)], [{}, {
                'force_batch_dim_to_mesh_dim': 0
            }]),
    16:
        get_solution_case(
            "4B", 1536, 32,
            [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]],
            [(1, 4), (1, 4), (1, 8)], [(4, 1), (4, 1), (8, 1)], [{
                'force_batch_dim_to_mesh_dim': 0
            }, {
                'force_batch_dim_to_mesh_dim': 0
            }, {}]),
    32:
        get_solution_case(
            "6.8B", 1536,
            32, [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15]],
            [(1, 8), (1, 8), (1, 8),
             (1, 8)], [(8, 1), (8, 1), (8, 1),
                       (8, 1)], [{
                           'force_batch_dim_to_mesh_dim': 0
                       }, {}, {}, {}]),
    64:
        get_solution_case("13B", 1520, 38,
                          [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                           [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23],
                           [24, 25, 26, 27, 28], [29, 30, 31, 32]], [(1, 8),
                                                                     (1, 8),
                                                                     (1, 8),
                                                                     (1, 8),
                                                                     (1, 8),
                                                                     (1, 8),
                                                                     (1, 8),
                                                                     (1, 8)],
                          [(8, 1), (1, 8), (8, 1), (1, 8), (8, 1), (8, 1),
                           (1, 8), (8, 1)], [{}, {
                               'force_batch_dim_to_mesh_dim': 0
                           }, {}, {
                               'force_batch_dim_to_mesh_dim': 0
                           }, {}, {}, {
                               'force_batch_dim_to_mesh_dim': 0
                           }, {}]),
}

# Grid search on hyperparameters
grid_search_auto_suite = {  # key = the number of gpus, value = a list of cases
    1: get_search_cases("250M", 1536, [24, 32]),
    2: get_search_cases("500M", 1536, [24, 32]),
    4: get_search_cases("1B", 1536, [24, 32]),
    8: get_search_cases("2B", 1536, [24, 32]),
    16: get_search_cases("4B", 1536, [24, 32]),
    32: (get_search_cases("6.8B", 1520, [38]) +
     get_search_cases("6.8B", 1512, [42])),
    64: get_search_cases("13B", 1520, [38]),
}
