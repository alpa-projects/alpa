
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
fixed_params = (False, True,  True, True) # FM, Remat, RS, AP
max_global_batch_size = 1024

default_overwrite_dict = {
    "auto_stage_construction_imbalance_tolerance": 1.0,
    "logical_mesh_search_space": "default",
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}

def get_auto_test_case(model_name, n_microbatches, num_layers, overwrite_global_config_dict=None):
    if overwrite_global_config_dict is None:
        overwrite_global_config_dict = default_overwrite_dict
    return [(max_global_batch_size, *moe_specs[model_name],
             *dummy_arguments, num_layer, n_microbatch, *fixed_params, overwrite_global_config_dict)
            for n_microbatch in n_microbatches
            for num_layer in num_layers]

paper_auto_moe_suite = {
2: (get_auto_test_case("380M", [16, 32, 64], [8]) +
    get_auto_test_case("690M", [16, 32, 64], [8])),
4: (get_auto_test_case("690M", [16, 32, 64], [8]) +
    get_auto_test_case("1.3B", [16, 32, 64], [8])),
8: (get_auto_test_case("1.3B", [16, 32, 64], [8]) +
    get_auto_test_case("2.4B", [16, 32, 64], [8])),
16: (get_auto_test_case("10B", [16, 32, 64], [8])),
32: (get_auto_test_case("27B", [16, 32, 64], [16])),
}

test_auto_moe_suite = {
1:  get_auto_test_case("380M", [64], [4]),
2:  get_auto_test_case("690M", [64], [4]),
4:  get_auto_test_case("1.3B", [64], [8]),
8:  get_auto_test_case("2.4B", [16], [8]),
16: get_auto_test_case("10B",  [32], [8]),
32: get_auto_test_case("27B",  [32], [8]),
}
