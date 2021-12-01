# suite for gpt benchmarking

gpt_specs = {
# Note: that head_size = hidden_size / #head
        # S，    H，    #L,   #head,  V,
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
fixed_params = (False, True, True, True) # FD,  Remat, RS, Auto layer & stage
max_global_batch_size = 1024


def get_auto_test_case(model_name, n_microbatches, num_layers):
    return [(max_global_batch_size, *gpt_specs[model_name],
             *dummy_arguments, num_layer, n_microbatch, *fixed_params)
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
2: (get_auto_test_case("350M", [32, 64, 128], [6]) +
    get_auto_test_case("760M", [64, 128, 256], [6])),
4: (get_auto_test_case("760M", [32, 64, 128], [6]) +
    get_auto_test_case("1.3B", [64, 128, 256], [6])),
8: (get_auto_test_case("1.3B", [32, 64, 128], [6]) +
    get_auto_test_case("2.7B", [64, 128, 256], [8])),
}

test_auto_gpt_suite = {
1: get_auto_test_case("125M", [64], [6]),
2: get_auto_test_case("350M", [64], [6]),
4: get_auto_test_case("760M", [64], [6]),
8: get_auto_test_case("1.3B", [64], [6]),
}
