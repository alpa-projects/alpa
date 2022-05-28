"""Benchmark suite for auto gpt."""
from benchmark.alpa.suite_paper_manual_gpt import gpt_specs

dummy_arguments = (0, 0, 0, 0) # LD0, LD1, PD0, PD1, not used for auto
fixed_params = (False, True, True) # FM, Remat, RS
max_global_batch_size = 1024

default_overwrite_dict = {
    "auto_stage_construction_imbalance_tolerance": 1.0,
    "logical_mesh_search_space": "all",
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database_osdi22_artifact.pkl",
    "submesh_choices_mode": "small_power_of_two",
}


def get_auto_test_case(model_name, n_microbatches, num_layers,
                       pipeline_stage_mode="auto_gpipe",
                       overwrite_global_config_dict=None):
    if overwrite_global_config_dict is None:
        overwrite_global_config_dict = default_overwrite_dict
    return [(max_global_batch_size, *gpt_specs[model_name],
             *dummy_arguments, num_layer, n_microbatch, *fixed_params,
             pipeline_stage_mode, overwrite_global_config_dict)
            for n_microbatch in n_microbatches
            for num_layer in num_layers]


artifact_search_e2e_gpt_suite = {
1: (get_auto_test_case("350M", [512], [1])),
2: (get_auto_test_case("760M", [64], [6])),
4:  get_auto_test_case("1.3B", [128], [6]),
8:  get_auto_test_case("2.6B", [128], [8]),
16: get_auto_test_case("6.7B", [64], [8]),
32: get_auto_test_case("15B", [128], [16]),
}


artifact_result_e2e_gpt_suite = {
1: get_auto_test_case("350M", [512], [1], "manual_gpipe", {
    "forward_stage_layer_ids": [[0]],
    "sub_physical_mesh_shapes": [(1, 1)],
    "sub_logical_mesh_shapes": [(1, 1)],
    "submesh_autosharding_option_dicts": [{}],
}),
2: get_auto_test_case("760M", [128], [6], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2], [3, 4, 5]],
    "sub_physical_mesh_shapes": [(1, 1)] * 2,
    "sub_logical_mesh_shapes": [(1, 1)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2,
}),
4: get_auto_test_case("1.3B", [128], [6], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2], [3, 4, 5]],
    "sub_physical_mesh_shapes": [(1, 2)] * 2,
    "sub_logical_mesh_shapes": [(2, 1)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0} for _ in range(2)],
}),
8: get_auto_test_case("2.6B", [128], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1], [2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 2), (1, 2), (1, 4)],
    "sub_logical_mesh_shapes": [(2, 1), (2, 1), (4, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {}, {}],
}),
16: get_auto_test_case("6.7B", [64], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 8)] * 2,
    "sub_logical_mesh_shapes": [(2, 4)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0} for _ in range(2)],
}),
32: get_auto_test_case("15B", [128], [16], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 8)] * 4,
    "sub_logical_mesh_shapes": [(2, 4)] * 4,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0} for _ in range(4)],
}),
}


artifact_result_inter_op_ablation_gpt_suite = {
16: # Ours
get_auto_test_case("6.7B", [64], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 8)] * 2,
    "sub_logical_mesh_shapes": [(2, 4)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0} for _ in range(2)],
}) + # Equal operator
get_auto_test_case("6.7B", [64], [8], "manual_gpipe", {
    "use_equal_eqn": True,
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 8)] * 2,
    "sub_logical_mesh_shapes": [(2, 4)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0} for _ in range(2)],
}) + # Equal layer
get_auto_test_case("6.7B", [64], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "sub_physical_mesh_shapes": [(1, 8)] * 2,
    "sub_logical_mesh_shapes": [(2, 4)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0} for _ in range(2)],
}),
}