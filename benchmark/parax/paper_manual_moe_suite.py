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
],

2: [
    # B,        model,              S',           LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,    remat, tie, auto , EP
],

4: [
    # B,        model,                  S',           LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,    remat, tie, auto , EP
],

8: [
    # B,  model,                                     LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,    remat, tie, auto , EP ,
],

16: [
],

32: [
]

}


paper_moe_suite = {
1: [
    # B,      model,                             LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,   (Remat, Tie, Auto-layer-slicing),

    # 1 GPUs, deepspeed max bs = 8, parax = 16
    (8,     *moe_specs["380M"],  8 * 1024 // 2,   1,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (32,    *moe_specs["380M"],  8 * 1024 // 2,   1,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (128,    *moe_specs["380M"], 8 * 1024 // 2,   1,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    (512,    *moe_specs["380M"], 8 * 1024 // 2,   1,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),
    (1024,    *moe_specs["380M"], 8 * 1024 // 2,   1,    1,    1,    1,   1,    128,  True,   *fixed_params,     1),
],

2: [
    # 2 GPUs, deepspeed max effective bs = 8
    # MP is always worse
    # ===============================
    # 380M model
    # B,      model,                             LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,   (Remat, Tie, Auto-layer-slicing),
    # (16,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    # (64,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    # (256,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    # (1024,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),
    #
    # (16,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    # (64,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    # (256,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     2),
    # (1024,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    64,  True,   *fixed_params,     2),

    # ================================
    # 690M model
    # DP = 2, EP = 1
    (16,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (64,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (256,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # DP = 2, EP = 2
    (16,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (64,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (256,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     2),
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    64,  True,   *fixed_params,     2),

    # DP = 1, MP = 2, EP = 1
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   1,    2,    1,    1,   1,    128,  True,   *fixed_params,     1),
],

4: [
    # B,      model,                             LD0,  LD1,  PD0,  PD1,  PP,  NB,   FD,   (Remat, Tie, Auto-layer-slicing),
    # ================================
    # 690M model
    # 4 GPUs, deepspeed max effective bs = 8, no MP
    # DP = 4, EP = 1
    (32,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (128,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    32,  True,   *fixed_params,     1),

    # DP = 4， EP = 2
    (32,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (128,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    32,  True,   *fixed_params,     2),

    # DP = 4， EP = 4
    (32,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (128,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    32,  True,   *fixed_params,     4),

    # DP = 2， EP = 1， MP = 2
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   2,    2,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # DP = 2, EP = 2, MP = 2
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   2,    2,    1,    1,   1,    64,  True,   *fixed_params,     2),

    # DP = 1， MP = 4， EP = 1
    # effective batch size = 16
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   1,    4,    1,    1,   1,    64,  True,   *fixed_params,     1),


    # ================================
    # 1.3B model
    # 4 GPUs, deepspeed max effective bs = 8, no MP
    # DP = 4, EP = 1
    (32,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (128,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    8,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # DP = 4, EP = 2
    (32,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (128,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    32,  True,   *fixed_params,     2),

    # DP = 4, EP = 4
    (32,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (128,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    1,    1,    1,   1,    32,  True,   *fixed_params,     4),

    # DP = 2, MP = 2, EP = 1
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   2,    2,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # DP =2, MP = 2, EP = 2
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   2,    2,    1,    1,   1,    64,  True,   *fixed_params,     2),

    # MP = 4
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   1,    4,    1,    1,   1,    128,  True,   *fixed_params,     1),

],

8: [
    # ================================
    # 1.3B model
    # DP = 8
    (64,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (256,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),

    # DP = 8, EP = 2
    (64,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (256,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    16,  True,   *fixed_params,     2),

    # DP = 8, EP = 4
    (64,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (256,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    16,  True,   *fixed_params,     4),

    # DP = 8, EP = 8
    (64,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (256,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    16,  True,   *fixed_params,     8),

    # DP = 4, MP = 2, EP = 4
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    2,    1,    1,   1,    32,  True,   *fixed_params,     4),

    # DP = 4, MP = 2, EP = 2
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    2,    1,    1,   1,    32,  True,   *fixed_params,     2),

    # DP = 4, MP = 2, EP = 1
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   4,    2,    1,    1,   1,    32,  True,   *fixed_params,     1),

    # DP = 2, MP = 4, EP = 2
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   2,    4,    1,    1,   1,    64,  True,   *fixed_params,     2),

    # DP = 2, MP = 4, EP = 1
    (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   2,    4,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # MP = 8: give up.. too  bad.
    # (1024,     *moe_specs["1.3B"],  8 * 1024 // 2,   1,    8,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # ================================
    # from now on, I give up MP.
    # 2.4B model
    # DP = 8, EP = 1
    (32,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (256,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    8,  True,   *fixed_params,     1),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    32,  True,   *fixed_params,     1),

    # DP = 8, EP = 2
    (64,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (256,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    16,  True,   *fixed_params,     2),

    # DP = 8, EP = 4
    (64,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (256,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    16,  True,   *fixed_params,     4),

    # DP = 8, ep = 8
    (64,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (256,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    1,    1,    1,   1,    16,  True,   *fixed_params,     8),

    # DP = 4, MP = 2， EP=4
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   4,    2,    1,    1,   1,    32,  True,   *fixed_params,     4),

],

16: [
    # ================================
    # 2.4B model, max(ep) = 8, otherwise extremely slow.

    # DP = 16, EP = 1
    (128,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (512,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     1),

    # DP = 16, EP = 2
    (128,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (512,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     2),

    # DP = 16, EP = 4
    (128,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (512,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     4),

    # DP = 16, EP = 8
    (128,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (512,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     8),

    # DP = 16， EP = 16:
    (128,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     16),
    (512,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     16),
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     16),

    # Note: MP * EP must <= 8
    # DP = 8, MP = 2, EP = 4
    (1024,     *moe_specs["2.4B"],  8 * 1024 // 2,   8,    2,    1,    1,   1,    16,  True,   *fixed_params,     4),


    # ================================
    # 4.5B model, max(ep) = 8, otherwise extremely slow.
    # DP = 16, EP = 2
    (128,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    2,  True,   *fixed_params,     2),
    (512,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     2),
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    16,  True,   *fixed_params,     2),

    # DP = 16, EP = 4
    (128,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (512,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     4),

    # DP = 16， EP = 8
    (128,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (512,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     8),

    # DP = 16， EP = 16
    (128,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     16),
    (512,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     16),
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     16),

    # DP = 8， MP = 2， EP = 4
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   8,    2,    1,    1,   1,    16,  True,   *fixed_params,     4),

    # DP = 4， MP = 4， EP = 4
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   4,    4,    1,    1,   1,    32,  True,   *fixed_params,     4),
],

32: [
    # ================================
    # 4.5B model, max(ep) = 8, otherwise extremely slow.
    # DP = 32, EP = 1, cannot accumulate..
    (32,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    # (256,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    # (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # DP = 32, EP = 2
    (256,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (2048,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     2),

    # DP = 32, EP = 4
    (256,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (2048,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     4),

    # DP = 32, EP = 8
    (256,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (2048,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     8),

    # DP = 32, EP = 16
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     16),

    # DP = 32, EP = 32
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     32),

    # EP * MP <= 8
    # DP = 8， MP = 4， EP = 2
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   8,    4,    1,    1,   1,    16,  True,   *fixed_params,     2),

    # DP = 8， MP = 2， EP = 4
    (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   16,    2,    1,    1,   1,    8,  True,   *fixed_params,     4),

    # ================================
    # 10B model
    # DP = 32, EP = 1 fails when batch size = 32
    # DP = 32, EP = 2 fails when batch size = 32
    # DP = 32 , EP = 4 fails when batch size = 32
    # DP = 32, EP = 8
    (128,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (512,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     8),

    # DP = 32, EP = 16
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     16),

    # EP * MP == 8
    # DP = 16, MP = 2, EP = 4
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   16,    2,    1,    1,   1,    16,  True,   *fixed_params,     4),


    # I skip the two below
    # DP = 8, MP = 4, EP = 2
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   8,    4,    1,    1,   1,    16,  True,   *fixed_params,     2),

    # DP = 4, MP = 8, EP = 1
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   4,    8,    1,    1,   1,    8,  True,   *fixed_params,     1),
],

64: [

]
}