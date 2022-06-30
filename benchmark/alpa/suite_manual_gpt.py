"""Benchmark suites for gpt with manual specifications."""

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads,
# NB = num_micro_batches, PM = parallel_mode
# 3D config = 3D parallel config (Data, Operator, Pipeline)
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,

gpt_specs = {
        #S，    H，     L,     head, V,
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

# Temporary debug suite
tmp_suite = {  # key = the number of gpus, value = a list of cases
    #B,         model,         NB,   PM,        RS,    Remat, 3D Config,  FM
1: [
    (16,  *gpt_specs["350M"],  1,    "manual", (True,  True,  (1, 1, 1),  True))
],

4: [
],

8: [
    #B,   S,     H     L,  #head, V,     NB,             RS,    Remat, 3D Config,  FM
    (128, 1024,  4096, 4,  32,    51200, 4,   "manual", (True,  True,  (4, 1, 2),  True)),
],

16: [
],

32: [
],
}


# Fast performance test on models with fewer layers
perf_test_fast_2d_suite = {
1: [
    #B,   S,     H     L,  #head, V,     NB,  PM,        RS,    Remat, 3D config,  FM
    (8,   1024,  1024, 4,  32,    51200, 1,   "manual", (False, True,  (1, 1, 1),  True))
],

8: [
    #B,   S,     H     L,  #head, V,     NB,             RS,    Remat, 3D Config,  FM
    (32,  1024,  4096, 4,  32,    51200, 1,   "manual", (True,  True,  (8, 1, 1),  True)),
    (128, 1024,  4096, 4,  32,    51200, 4,   "manual", (True,  True,  (8, 1, 1),  True)),
],
}

# Performance test on normal models
perf_test_suite = {
    #B,         model,         NB,   PM,        RS,    Remat, 3D Config,  FM
1: [
    (16,  *gpt_specs["350M"],  1,    "manual", (True,  True,  (1, 1, 1),  True))
],

8: [
    (32,  *gpt_specs["2.6B"],  4,    "manual", (True,  True,  (2, 2, 2),  True))

],

64: [
    (1024, *gpt_specs["39B"],  1024, "manual", (True,  True,  (1, 4, 16), True))
],
}
