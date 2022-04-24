"""Benchmark suite for gpt."""

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads, LD0 = logical_mesh_dimension_0, LD1 = logical_mesh_dimension_1,
# PD0 = physical_mesh_dimension_0, PD1 = physical_mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping, Remat = use_rematerialization
# RS = prefer_reduce_scatter, Stage = pipeline_stage_mode

gpt_specs = {
        #S，    H，    L,     head, V,
"125M": (1024,  768,   12,    12,   51200,),
"350M": (1024,  1024,  24,    16,   51200,),
"760M": (1024,  1536,  24,    16,   51200,),
"1.3B": (1024,  2048,  24,    32,   51200,),
"2.6B": (1024,  2560,  32,    32,   51200,),
"6.7B": (1024,  4096,  32,    32,   51200,),
"15B":  (1024,  5120,  48,    40,   51200,),
"39B":  (1024,  8192,  48,    64,   51200,),
"76B":  (1024,  10240, 60,    80,   51200,),
}

_ = None

              # Remat, RS,   Stage,                 overwrite_global_config_dict
fixed_params = (True,  True, "uniform_layer_gpipe", None)
max_global_batch_size = 1024

# Temporary debug suite
tmp_gpt_suite = {  # key = the number of gpus, value = a list of cases
    #B,         model,         LD0, LD1, PD0, PD1, PP, NB,  FM,    ...
1: [
    (32,  *gpt_specs["350M"],  1,   1,   1,   1,   1,  1,   False, *fixed_params),
],

4: [
    (256,  *gpt_specs["760M"], 4,   1,   1,   4,   1,  32,  True,  *fixed_params),
],

8: [
    (32,  *gpt_specs["2.6B"],  2,   2,   1,   4,   2,  4,   False, *fixed_params),
],

16: [
    (1024, *gpt_specs["6.7B"], 2,   1,   1,   2,   8,  256, True,  *fixed_params),
],

32: [
    (4,  *gpt_specs["15B"],    1,   8,   1,   8,   4,  1,   True,  *fixed_params),
],
}

# Fast performance test on models with fewer layers
fast_perf_test_gpt_suite = {
1: [
    #B,   S,     H     L,  #head, V,     LD0, LD1, _, _,  PP,  NB, FM,   Remat, RS,    _  _
    (8,  1024,  1024,  4,  32,    51200, 1,   1,   _, _,  1,   1,  True, True,  False, _, _),
],

8: [
    #B,   S,     H     L,  #head, V,     LD0, LD1, _, _,  PP,  NB, FM,   Remat, RS,    _  _
    (16,  1024,  8192,  4,  32,   51200, 1,   8,   _, _,  1,   1,  True, True,  True,  _, _),
    (16,  1024,  8192,  4,  32,   51200, 2,   4,   _, _,  1,   1,  True, True,  True,  _, _),
    (16,  1024,  8192,  4,  32,   51200, 8,   1,   _, _,  1,   1,  True, True,  True,  _, _),
],
}


# Performance test on normal models
perf_test_gpt_suite = {
    #B,         model,         LD0, LD1, PD0, PD1, PP, NB,  FM,    ...
1: [
    (16,  *gpt_specs["350M"],  1,   1,   1,   1,   1,  1,   True,  *fixed_params),
],

8: [
    (32,  *gpt_specs["2.6B"],  2,   2,   1,   4,   2,  4,   False, *fixed_params),
],
}
