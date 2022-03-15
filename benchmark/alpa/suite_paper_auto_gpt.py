"""Benchmark suite for auto gpt."""
from suite_paper_manual_gpt import gpt_specs

dummy_arguments = (0, 0, 0, 0) # LD0, LD1, PD0, PD1, not used for auto
fixed_params = (False, True, True) # FM, Remat, RS
max_global_batch_size = 1024

default_overwrite_dict = {
    "auto_stage_construction_imbalance_tolerance": 1.0,
    "logical_mesh_search_space": "all",
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database_4_node_20220111.pkl",
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


paper_auto_gpt_suite = {
2: (get_auto_test_case("760M", [32, 64, 128, 256], [6]) +
    get_auto_test_case("760M", [32, 64], [12])),
4: (get_auto_test_case("1.3B", [32, 64, 128], [6]) +
    get_auto_test_case("1.3B", [32, 64], [12])),
8: (get_auto_test_case("2.6B", [64, 128, 256], [8]) +
    get_auto_test_case("2.6B", [64, 128], [16])),
16: get_auto_test_case("6.7B", [32, 64, 128, 256], [8]),
32: get_auto_test_case("15B", [64, 128, 256, 512], [16]),
64: get_auto_test_case("39B", [128, 256, 512, 1024], [8]),
}


tmp_auto_gpt_suite = {
1: get_auto_test_case("125M", [64], [6]),
2: get_auto_test_case("350M", [64], [6]),
4: get_auto_test_case("760M", [64], [6], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1, 2], [3, 4, 5]],
    "sub_physical_mesh_shapes": [(1, 2)] * 2,
    "sub_logical_mesh_shapes": [(2, 1)] * 2,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2,
}),
8: get_auto_test_case("2.6B", [128], [8]),
16: get_auto_test_case("6.7B", [256], [8]),
32: get_auto_test_case("15B", [256], [16], "manual_gpipe", {
    "forward_stage_layer_ids": [[2 * i, 2 * i + 1] for i in range(8)],
    "sub_physical_mesh_shapes": [(1, 4)] * 8,
    "sub_logical_mesh_shapes": [(4, 1), (4, 1), (4, 1), (4, 1), (1, 4), (4, 1), (1, 4), (1, 4)],
    "submesh_autosharding_option_dicts": [{}, {}, {}, {}, {'force_batch_dim_to_mesh_dim': 0}, {},
                                          {'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}]}),
64: get_auto_test_case("39B", [256], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[i] for i in range(8)],
    "sub_physical_mesh_shapes": [(1, 8)] * 8,
    "sub_logical_mesh_shapes": [(1, 8)] * 8,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 8,
})
}
