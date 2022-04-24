"""Benchmark suites for moe with manual specifications."""

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads, S_ = expert_group_size, E = expert_number,
# LD0 = logical_mesh_dimension_0, LD1 = logical_mesh_dimension_1,
# PD0 = physical_mesh_dimension_0, PD1 = physical_mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping, Remat = use_rematerialization
# RS = prefer_reduce_scatter, Stage = pipeline_stage_mode

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

#               Remat, RS,   Stage
fixed_params = (True,  True, "uniform_layer_gpipe")

_ = None

# Temporary debug suite
tmp_suite = {  # key = the number of gpus, value = a list of cases
1: [
    # B,      model,                 LD0, LD1, PD0, PD1, PP, NB, FM,    ...,            EP (for deepspeed)
    (8,  *moe_specs["380M"],  2048,  1,   1,   1,   1,   1,  1,  False, *fixed_params,  1),
],

4: [
],

8: [
    (16, *moe_specs["1.3B"],  2048,  1,   4,   1,   4,   2,  1,  False, *fixed_params, 1),
],

16: [
     # verify cost model vs. profiling
    (1024, *moe_specs["10B"], 2048,  2,   8,   2,   8,   1,  32, True,  True, True, _, _),
],

32: [
],
}

# Fast performance test on models with fewer layers
perf_test_fast_2d_suite = {
1: [
    #B,  S,    H     L,  #head, V,     E,  S_,   LD0, LD1, _, _,  PP,  NB, FM,    Remat, RS,    _, _
    (8,  1024, 1024, 8,  32,    25600, 8,  1024, 1,   1,   _, _,  1,   1,  True,  True,  True,  _, _),
],

8: [
    #B,  S,    H     L,  #head, V,     E,  S_,   LD0, LD1, _, _,  PP,  NB, FM,    Remat, RS,    _, _

    # DEBUG: #all-to-all should be the same for the following mixed logical mesh shape cases
    (16, 1024, 1024, 4,  32,    25600, 32, 1024, 8,   1,   _, _,  1,   1,  False, False, True,  _, _),
    (16, 1024, 1024, 4,  32,    25600, 32, 1024, 4,   2,   _, _,  1,   1,  False, False, True,  _, _),
    (16, 1024, 1024, 4,  32,    25600, 32, 1024, 2,   4,   _, _,  1,   1,  False, False, True,  _, _),
],
}
