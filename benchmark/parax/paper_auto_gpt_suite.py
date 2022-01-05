# suite for gpt benchmarking

gpt_specs = {
# Note: that head_size = hidden_size / #head
        # S，    H，    #L,   #head,  V,
"TEST": (128,   128,   2,     8,    512,   ),
"125M": (1024,  768,   12,    12,   51200, ),
"350M": (1024,  1024,  24,    16,   51200, ),
"760M": (1024,  1536,  24,    16,   51200, ),
"1.3B": (1024,  2048,  24,    32,   51200, ),
"2.7B": (1024,  2560,  32,    32,   51200, ),
"6.7B": (1024,  4096,  32,    32,   51200, ),
"15B":  (1024,  5120,  48,    40,   51200, ),
"39B":  (1024,  8192,  48,    64,   51200, ),
"76B":  (1024,  10240, 60,    80,   51200, ),
}

dummy_arguments = (0, 0, 0, 0) # LD0, LD1, PD0, PD1, not used for auto
fixed_params = (False, True, True) # FM, Remat, RS
max_global_batch_size = 1024

default_overwrite_dict = {
    "auto_stage_construction_imbalance_tolerance": 1e5,
    "logical_mesh_search_space": "all",
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
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
# 1: (get_auto_test_case("125M", [16, 32, 64, 128, 256], [6]) +
#     get_auto_test_case("350M", [32, 64, 128, 256, 512], [6])),
# 2: (get_auto_test_case("350M", [16, 32, 64, 128, 256], [6]) +
#     get_auto_test_case("760M", [32, 64, 128, 256, 512], [6])),
# 4: (get_auto_test_case("760M", [16, 32, 64, 128, 256], [6]) +
#     get_auto_test_case("1.3B", [32, 64, 128, 256, 512], [6])),
# 8: (get_auto_test_case("1.3B", [16, 32, 64, 128, 256], [6]) +
#     get_auto_test_case("2.7B", [32, 64, 128, 256, 512], [8])),
2: (get_auto_test_case("350M", [16, 32, 64, 128], [6]) +
    get_auto_test_case("760M", [32, 64, 128, 256], [6]) +
    get_auto_test_case("350M", [32], [12]) +
    get_auto_test_case("760M", [64], [12])),
4: (get_auto_test_case("760M", [16, 32, 64], [6]) +
    get_auto_test_case("1.3B", [32, 64, 128], [6]) +
    get_auto_test_case("760M", [32], [12]) +
    get_auto_test_case("1.3B", [64], [12])),
8: (get_auto_test_case("1.3B", [64, 128, 256], [6]) +
    get_auto_test_case("2.7B", [64, 128, 256], [8]) +
    get_auto_test_case("1.3B", [16], [12]) +
    get_auto_test_case("2.7B", [64], [16])),
16: (# get_auto_test_case("2.7B", [16, 32, 64, 128], [8]) +
     get_auto_test_case("6.7B", [32, 64, 128, 256], [32, 34])
     # get_auto_test_case("2.7B", [64], [16]) +
     # get_auto_test_case("6.7B", [128], [16])
     ),
32: (# get_auto_test_case("6.7B", [16, 32, 64, 128], [8]) +
     # get_auto_test_case("15B", [64, 128, 256, 512], [8]) +
     get_auto_test_case("6.7B", [64, 128], [32, 34]) +
     get_auto_test_case("15B", [128, 256, 512], [32, 34])),
}

test_auto_gpt_suite = {
1: get_auto_test_case("125M", [64], [6]),
2: get_auto_test_case("350M", [64], [6]),
4: get_auto_test_case("760M", [64], [6]),
8: get_auto_test_case("2.7B", [128], [8]),
# 8: get_auto_test_case("TEST", [2], [2]),
# 16: get_auto_test_case("2.7B", [16], [8]),
16: get_auto_test_case("6.7B", [256], [8]),
}

result_auto_gpt_suite = {
32: get_auto_test_case("6.7B", [64], [32], "manual_gpipe", {
    "forward_stage_layer_ids": [[4 * i + j for j in range(4)] for i in range(8)],
    "sub_physical_mesh_shapes": [(1, 4)] * 8,
    "sub_logical_mesh_shapes": [(4, 1)] * 8,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 8,
})
}
