# suites for moe benchmarking

moe_specs = {
#         S,    H,      L,     #head,     V,    S_, E
"380M":  (1024, 768,    8,     16,      32000,  1,  8,),
"690M":  (1024, 768,    8,     16,      32000,  1,  16,),
"1.3B":  (1024, 768,    16,    16,      32000,  1,  16,),
"2.4B":  (1024, 1024,   16,    16,      32000,  1,  16,),
"4.5B":  (1024, 1024,   16,    16,      32000,  1,  32,),
"10B":   (1024, 1024,   24,    16,      32000,  1,  48,),
"18B":   (1024, 1024,   32,    16,      32000,  1,  64,),
"35B":   (1024, 1024,   32,    16,      32000,  1,  128,),
}

#               Remat, Tie, Auto-layer-slicing
fixed_params = (True,  False,  False)


test_moe_suite = {

1: [
    # B,  model,              LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,    remat, tie, auto , EP ,
    # (16, *moe_specs["380M"],   1,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    (4, *moe_specs["380M"],   1,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    # (128, *moe_specs["380M"],   1,    1,    1,    1,   1,    16,  True,    *fixed_params,     1),
]

}


paper_moe_suite = {
1: [
    # B,  model,              LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,   (Remat, Tie, Auto-layer-slicing),
    (32, *moe_specs["380M"],   1,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),
    (64, *moe_specs["380M"],   1,    1,    1,    1,   1,    1,  True,   *fixed_params,  1),

],

2: [

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