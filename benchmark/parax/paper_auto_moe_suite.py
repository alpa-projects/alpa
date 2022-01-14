"""Benchmark suite for auto moe."""

moe_specs = {
#         S,    H,      L,     #head,   V,      E,   S_
"380M":  (1024, 768,    8,     16,      32000,  8,   2048),
"690M":  (1024, 768,    8,     16,      32000,  16,  2048),
"1.3B":  (1024, 768,    16,    16,      32000,  16,  2048),
"2.4B":  (1024, 1024,   16,    16,      32000,  16,  2048),
"10B":   (1024, 1536,   16,    16,      32000,  32,  2048),
"27B":   (1024, 2048,   16,    16,      32000,  48,  2048),
"70B":   (1024, 2048,   32,    16,      32000,  64,  2048),
"140B":  (1024, 2048,   32,    16,      32000,  128, 2048),
}

_ = None

dummy_arguments = (1, 1, 0, 0) # LD0, LD1, PD0, PD1, not used for auto
fixed_params = (False, True, True) # FM, Remat, RS
max_global_batch_size = 1024

default_overwrite_dict = {
    "auto_stage_construction_imbalance_tolerance": 1.0,
    "logical_mesh_search_space": "all",
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}

def get_auto_test_case(model_name, n_microbatches, num_layers,
                       pipeline_stage_mode="auto_gpipe",
                       overwrite_global_config_dict=None):
    if overwrite_global_config_dict is None:
        overwrite_global_config_dict = default_overwrite_dict
    return [(max_global_batch_size, *moe_specs[model_name],
             *dummy_arguments, num_layer, n_microbatch, *fixed_params,
             pipeline_stage_mode, overwrite_global_config_dict)
            for n_microbatch in n_microbatches
            for num_layer in num_layers]

paper_auto_moe_suite = {
2: (get_auto_test_case("690M", [16, 32, 64], [8])),
4: (get_auto_test_case("1.3B", [16, 32, 64], [8])),
8: (get_auto_test_case("2.4B", [16, 32, 64], [8])),
16: (get_auto_test_case("10B", [16, 32, 64], [8])),
32: (get_auto_test_case("27B", [16, 32, 64], [4, 8, 16])),
}

test_auto_moe_suite = {
1:  get_auto_test_case("380M", [64], [4]),
2:  get_auto_test_case("690M", [64], [4]),
#4:  get_auto_test_case("1.3B", [64], [8]),
4: get_auto_test_case("1.3B", [64], [8], "manual_gpipe", {
    "forward_stage_layer_ids": [[0, 1], [2, 3], [4, 5], [6, 7]],
    "sub_physical_mesh_shapes": [(1, 1)] * 4,
    "sub_logical_mesh_shapes": [(1, 1)] * 4,
    "submesh_autosharding_option_dicts": [{}] * 4,
}),
8:  get_auto_test_case("2.4B", [16], [8]),
16: get_auto_test_case("10B",  [32], [8]),
32: get_auto_test_case("27B",  [32], [8]),
}
