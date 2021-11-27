# suites for moe benchmarking

moe_specs = {
#         S,    H,      L,     #head,     V,     E
"380M":  (1024, 768,    8,     16,      32000,   8,),
"690M":  (1024, 768,    8,     16,      32000,   16,),
"1.3B":  (1024, 768,    16,    16,      32000,   16,),
"2.4B":  (1024, 1024,   16,    16,      32000,   16,),
"4.5B":  (1024, 1024,   16,    16,      32000,   32,),
"10B":   (1024, 1024,   24,    16,      32000,   48,),
"18B":   (1024, 1024,   32,    16,      32000,   64,),
"35B":   (1024, 1024,   32,    16,      32000,   128,),
}

#               Remat, Tie, Auto-layer-slicing
fixed_params = (True,  False,  False)


test_moe_suite = {

1: [
    # B,  model,              S',       LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,    remat, tie, auto , EP

    # (32, *moe_specs["380M"],  16 * 1024,   1,    1,    1,    1,   1,    2,  True,   *fixed_params,     1),
    (8, *moe_specs["380M"],  8 * 1024 // 2,  1,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    # (16, *moe_specs["380M"],  16 * 1024,   1,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
],

2: [
    # B,  model,              LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,    remat, tie, auto , EP ,
    # (16, *moe_specs["380M"],   1,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    # (8, *moe_specs["380M"],   1,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    # (16, *moe_specs["380M"],   2,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    (8, *moe_specs["380M"],   1,    2,    1,    1,   1,    1,  True,   *fixed_params,  4),
    # (16, *moe_specs["380M"],   1,    2,    1,    1,   1,    1,  True,   *fixed_params,  2),
],

4: [
    # B,  model,              LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,    remat, tie, auto , EP ,
    # (16, *moe_specs["380M"],   1,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    # (8, *moe_specs["380M"],   1,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    # (16, *moe_specs["380M"],   2,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    (32, *moe_specs["380M"],   4,    1,    1,    1,   1,    1,  True,   *fixed_params,  4),
    (32, *moe_specs["380M"],   4,    1,    1,    1,   1,    1,  True,   *fixed_params,  2),
    (32, *moe_specs["380M"],   4,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    # (64, *moe_specs["380M"],   1,    4,    1,    1,   1,    1,  True,   *fixed_params,  4),
]

}


paper_moe_suite = {
1: [
    # B,      model,                        LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,   (Remat, Tie, Auto-layer-slicing),

    # 1 GPUs, deepspeed max bs = 8, parax = 16
    (8,     *moe_specs["380M"],  8 * 1024 // 2,   1,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (32,    *moe_specs["380M"],  8 * 1024 // 2,   1,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (128,    *moe_specs["380M"], 8 * 1024 // 2,   1,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    (512,    *moe_specs["380M"], 8 * 1024 // 2,   1,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),
    (1024,    *moe_specs["380M"], 8 * 1024 // 2,   1,    1,    1,    1,   1,    128,  True,   *fixed_params,     1),
],

2: [
    # 2 GPUs, deepspeed max bs = ?
    (8,     *moe_specs["380M"],  8 * 1024,   1,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
],

4: [

],

8: [

],

16: [

],

32: [

],

64: [

]
}