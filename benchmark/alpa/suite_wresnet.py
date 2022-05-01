"""Suites for wresnet benchmarking."""
_ = None

# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor,
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping,
# RS = prefer_reduce_scatter, Remat = use_rematerialization
# LS = logical_mesh_search_space

wresnet_specs = {
    #    I,   L,  C,   W,  dtype,  
"250M": (224, 50, 160, 2,  "fp32"), 
"500M": (224, 50, 224, 2,  "fp32"), 
"1B":   (224, 50, 320, 2,  "fp32"), 
"2B":   (224, 50, 448, 2,  "fp32"), 
"4B":   (224, 50, 640, 2,  "fp32"), 
"6.8B": (224, 50, 320, 16, "fp32"), 
"12.6B":(224, 50, 320, 32, "fp32"),
}

default_overwrite_dict = {
    "auto_stage_construction_imbalance_tolerance": 0.4,
    "submesh_choices_mode": "small_power_of_two",
}


def get_auto_test_case(model_name,
                       n_microbatches,
                       max_global_batch_size,
                       logical_mesh_search_space="single_node_model_parallel",
                       overwrite_global_config_dict=None):
    # FD,    RS,   Remat
    fixed_params = [False, True, True]
    if overwrite_global_config_dict is None:
        overwrite_global_config_dict = default_overwrite_dict
    return [
        (max_global_batch_size, *wresnet_specs[model_name], n_microbatch,
         *fixed_params, logical_mesh_search_space, overwrite_global_config_dict)
        for n_microbatch in n_microbatches
    ]

# Performance test with shard parallel
tmp_suite = { # key = the number of gpus, value = a list of cases
8: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FM,    RS,    Remat, other
    (64,   224, 50,  320, 2, "fp32", 1,  8,  1,  False, True,  False, _),
],
}

# Performance test with shard parallel
perf_test_2d_suite = { # key = the number of gpus, value = a list of cases
1: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FM,    RS,    Remat, other
    (16,   224, 50,  192, 2, "fp32", 1,  1,  1,  False, True,  False, _),
],

4 : [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FM,    RS,    Remat, other
    (32,   224, 50,  320, 2, "fp32", 1,  4,  1,  False, False, False, _),
],

8: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FM,    RS,    Remat, other
    (64,   224, 50,  320, 2, "fp32", 1,  8,  1,  False, False, False, _),
],
}


# Performance test with search solutions found for p3.16xlarge
perf_test_auto_suite = {
1: get_auto_test_case("250M", [24], 1536, overwrite_global_config_dict={
    "strategy": "shard_parallel"
}),
2: get_auto_test_case("500M", [24], 1536, overwrite_global_config_dict={
    "strategy": "shard_parallel"
}),
4: get_auto_test_case("1B", [24], 1536, overwrite_global_config_dict={
    "strategy": "shard_parallel"
}),
8: get_auto_test_case("2B", [24], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 4)],
    "sub_logical_mesh_shapes": [(4, 1), (1, 4)],
    "submesh_autosharding_option_dicts": [{}, {'force_batch_dim_to_mesh_dim': 0}]
}),
16: get_auto_test_case("4B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 4), (1, 8)],
    "sub_logical_mesh_shapes": [(4, 1), (4, 1), (8, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}, {}]
}),
32: get_auto_test_case("6.8B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 8), (1, 8), (1, 8), (1, 8)],
    "sub_logical_mesh_shapes": [(8, 1), (8, 1), (8, 1), (8, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {}, {}, {}],
}),
64: get_auto_test_case("12.6B", [38], 1520, "single_node_model_parallel", {
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27, 28], [29, 30, 31, 32]],
    "sub_physical_mesh_shapes": [(1, 8), (1, 8), (1, 8), (1, 8), (1, 8), (1, 8), (1, 8), (1, 8)],
    "sub_logical_mesh_shapes": [(8, 1), (1, 8), (8, 1), (1, 8), (8, 1), (8, 1), (1, 8), (8, 1)],
    "submesh_autosharding_option_dicts": [{}, {'force_batch_dim_to_mesh_dim': 0}, {}, {'force_batch_dim_to_mesh_dim': 0}, {}, {}, {'force_batch_dim_to_mesh_dim': 0}, {}]
})
}


# Grid search on hyperparameters
grid_search_auto_suite = {  # key = the number of gpus, value = a list of cases
1: get_auto_test_case("250M", [24, 32], 1536),

2: get_auto_test_case("500M", [24, 32], 1536),

4: get_auto_test_case("1B", [24, 32], 1536),

8: get_auto_test_case("2B", [24, 32], 1536),

16: get_auto_test_case("4B", [24, 32], 1536),

32: (get_auto_test_case("6.8B", [38], 1520) +
     get_auto_test_case("6.8B", [42], 1512)),

64: get_auto_test_case("12.6B", [38], 1520),
}
