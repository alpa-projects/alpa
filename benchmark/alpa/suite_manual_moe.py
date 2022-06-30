"""Benchmark suites for moe with manual specifications."""

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads, S_ = expert_group_size, E = expert_number,
# NB = num_micro_batches, PM = parallel_mode
# 3D config = 3D parallel config (Data, Operator, Pipeline)
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,

moe_specs = {
#         S,    H,      L,     head,    V,       E
"380M":  (1024, 768,    8,     16,      32000,   8,),
"690M":  (1024, 768,    8,     16,      32000,   16,),
"1.3B":  (1024, 768,    16,    16,      32000,   16,),
"2.4B":  (1024, 1024,   16,    16,      32000,   16,),
"10B":   (1024, 1536,   16,    16,      32000,   32,),
"27B":   (1024, 2048,   16,    16,      32000,   48,),
"70B":   (1024, 2048,   32,    16,      32000,   64,),
"140B":  (1024, 2048,   32,    16,      32000,   128,),
}


_ = None

# Temporary debug suite
tmp_suite = {  # key = the number of gpus, value = a list of cases
1: [
    #B,         model,         S_,    NB,  PM,       RS,    Remat, 3D Config,  FM
    (8,  *moe_specs["380M"],   2048,  1,  "manual", (True,  True,  (1, 1, 1),  False))
],

4: [
],

8: [
    (16,   *moe_specs["1.3B"], 2048,  1,  "manual", (True,  True,  (1, 4, 2),  False))
],

16: [
     # verify cost model vs. profiling
    (1024, *moe_specs["10B"],  2048,  32, "manual", (True,  True,  (2, 8, 1),  True))
],

32: [
],
}

# Fast performance test on models with fewer layers
perf_test_fast_2d_suite = {
1: [
    #B,  S,    H     L,  #head, V,     E,  S_,   NB, PM         Remat, RS,    3D Config,  FM
    (8,  1024, 1024, 8,  32,    25600, 8,  1024, 1,  "manual", (True,  True,  (1, 1, 1),  True)),
],

8: [
    #B,  S,    H     L,  #head, V,     E,  S_,   NB, PM,        Remat, RS,    3D Config,  FM

    (16, 1024, 1024, 4,  32,    25600, 32, 1024, 1,  "manual", (False, True,  (8, 1, 1),  False)),
    (16, 1024, 1024, 4,  32,    25600, 32, 1024, 1,  "manual", (False, True,  (4, 2, 1),  False)),
    (16, 1024, 1024, 4,  32,    25600, 32, 1024, 1,  "manual", (False, True,  (2, 4, 1),  False)),
],
}
