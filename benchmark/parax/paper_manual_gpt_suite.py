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

"6.7B-half": (1024,  4096,  16,    32,   51200, ),
}


fixed_params = (True, True, False, None)
max_global_batch_size = 1024

test_gpt_suite = {
    #B,         model,         LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    AP
1: [
],

4: [
    (256,  *gpt_specs["760M"], 4,   1,   1,   4,   1,  32,  True,  *fixed_params),
],

8: [
    #B,         model,         LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    AP
    # 222 performance case. Ours: 37 TFLOPS. Megatron: 38 TFLOPS.
    (32,  *gpt_specs["2.7B"],  2,   2,   1,   4,   2,  4,    False,  *fixed_params),

    # 142 performance case.
    #(16,  *gpt_specs["6.7B-half"],  1,   4,   1,   4,   2,  1,    False,  *fixed_params),
    #(16,  1024, 2048, 8, 32, 51200, 1,   4,   1,   4,   2,  1,   True,  *fixed_params),
],

16: [
],

32: [
]

}


paper_gpt_suite = {
    #B,         model,         LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    AP
1: [
    # 125M
    (2,   *gpt_specs["125M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (4,   *gpt_specs["125M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (8,   *gpt_specs["125M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (16,  *gpt_specs["125M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["125M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["125M"],  1,   1,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["125M"],  1,   1,   1,   1,   1, 4,    1,  *fixed_params),

    # 350M
    (2,   *gpt_specs["350M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (4,   *gpt_specs["350M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (8,   *gpt_specs["350M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (16,  *gpt_specs["350M"],  1,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["350M"],  1,   1,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["350M"],  1,   1,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["350M"],  1,   1,   1,   1,   1, 8,    1,  *fixed_params),
],

2: [
    # 350M, max_bs = 8 per gpu (whole model)
    # DP
    (16,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  1,    True,  *fixed_params),
    (64,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  4,   1,  *fixed_params),
    (256,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  8,   True,  *fixed_params),

    # parax-only
    (256,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  4,   True,  *fixed_params),
    (256,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  8,   True,  *fixed_params),

    # MP
    (16,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  16,    False,  *fixed_params),

    # PP
    (16,  *gpt_specs["350M"],  1,   1,   1,   1,   2,  1,    False,  *fixed_params),
    (32,  *gpt_specs["350M"],  1,   1,   1,   1,   2,  2,    False,  *fixed_params),
    (64,  *gpt_specs["350M"],  1,   1,   1,   1,   2,  4,    False,  *fixed_params),
    (128,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  8,    False,  *fixed_params),

    #parax
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  16,    False,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  32,    False,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  64,    False,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  128,    False,  *fixed_params),

    # 760M, cannot train even with bs = 1 per gpu
    # DP all OOM on megatron

    # parax-only
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   1,  64,    True,  *fixed_params),
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   1,  32,    True,  *fixed_params),

    # MP, each 1/2 model can have max batch_size = 8
    (16,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  4,    False,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,  2,   1,   1,   1,  8,    False,  *fixed_params),

    # parax-only
    (512,  *gpt_specs["760M"],  1,  2,   1,   1,   1,  32,    False,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  2,   1,   1,   1,  64,    False,  *fixed_params),

    # PP, each 1/2 model can have maax batch_size = 8
    (16,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  1,    False,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  2,    False,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  4,    False,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  4,    False,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  8,    False,  *fixed_params),

    # parax-only
    (128,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  8,    False,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  16,    False,  *fixed_params),
    (256,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  16,    False,  *fixed_params),
    (256,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  32,    False,  *fixed_params),
    (512,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  32,    False,  *fixed_params),
    (512,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  64,    False,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  64,    False,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  128,    False,  *fixed_params),
],

4: [
    # 760M, max degree of DP = 2
    (32,  *gpt_specs["760M"],  2,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  2,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["760M"], 2,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (256,  *gpt_specs["760M"], 2,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (512,  *gpt_specs["760M"], 2,   2,   1,   1,   1,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"], 2,   2,   1,   1,   1,  32,    1,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["760M"], 4,   1,   1,   1,   1,  16,    True,  *fixed_params),


    # MP-only, max per-gpu batch size = 8, but actullay many cases fail.
    (32,  *gpt_specs["760M"],  1,   4,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   4,   1,   1,   1,  2,    1,  *fixed_params), # OOM
    (128,  *gpt_specs["760M"],  1,   4,   1,   1,   1, 4,    1,  *fixed_params), # OOM
    (256,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  8,    1,  *fixed_params), # OOM
    (512,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  32,    1,  *fixed_params), # OOM

    #parax-only
    (1024,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  32,    False,  *fixed_params),

    # PP-only
    (32,  *gpt_specs["760M"],  1,   1,   1,   1,   4,  1,    1,  *fixed_params), # OOM
    (64,  *gpt_specs["760M"],  1,   1,   1,   1,   4,  2,    1,  *fixed_params), # OOM
    (128,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  4,    1,  *fixed_params), # OOM
    (128,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  8,    1,  *fixed_params), # OOM
    (256,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  16,    1,  *fixed_params), # OOM
    (512,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  32,    1,  *fixed_params), # OOM
    (1024,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  64,    1,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  64,    True,  *fixed_params),

    # PP + DP
    (32,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  64,    1,  *fixed_params),

    # parax-only

    (1024,  *gpt_specs["760M"],  2,   1,   1,   2,   2,  32,    True,  *fixed_params),

    # PP + MP
    # max per-gpu batch = 4
    (32,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  64,    1,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["760M"],  1,   2,   1,   2,   2,  32,    False,  *fixed_params),

    # ====================================== 1.3B model

    # parax-only
    # Megatron cannot do DP-only
    (1024,  *gpt_specs["1.3B"],  4,   1,   1,   4,   1,  32,    True,  *fixed_params),

    # DP + MP
    (32,  *gpt_specs["1.3B"],  2,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  16,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  32,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  128,    1,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["1.3B"], 2,   2,   1,   4,   1,  32,    False,  *fixed_params),

    # MP-only, max per-gpu batch size = 4, parax skips MP-only
    (32,  *gpt_specs["1.3B"],  1,   4,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  1,   4,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  1,   4,   1,   1,   1, 8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  1,  4,   1,   1,   1,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  1,  4,   1,   1,   1,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  1,  4,   1,   1,   1,  64,    1,  *fixed_params),


    # PP-only, max per-gpu batch size = 2
    (32,  *gpt_specs["1.3B"],  1,   1,   1,   1,   4,  4,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  1,   1,   1,   1,   4,  8,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  16,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  32,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  64,    1,  *fixed_params),
    # parax
    (1024,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  128,    True,  *fixed_params),


    # PP + DP, max per-gpu batch size = 2
    (32,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  8,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  16,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  128,    1,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["1.3B"],  2,  1,   1,   2,   2,  64,    True,  *fixed_params),

    # PP + MP, parax skips PP+MP
    # max per-gpu batch = 4
    (32,  *gpt_specs["1.3B"],  1,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  1,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  1,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  1,  2,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  1,  2,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  1,  2,   1,   1,   2,  64,    1,  *fixed_params),
],

8: [
    # 1.3B model

    # DP = 8, Parax-only
    (1024,  *gpt_specs["1.3B"],  8,   1,   1,   8,   1,   8,    True,  *fixed_params),


    # DP 4 + MP 2, max per-gpu bs = 2
    (16,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  64,    False,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  8,    False,  *fixed_params),

    # DP 4 + PP 2, max per-gpu bs = 2
    (16,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  1,    1,  *fixed_params),
    (32,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  32,    1,  *fixed_params),
    # parax
    (1024,  *gpt_specs["1.3B"],  4,   1,   1,   4,   2,  64,    True,  *fixed_params),

    # DP2 + MP4
    (32,  *gpt_specs["1.3B"],  2,   4,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   4,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  2,   4,   1,   1,   1,  4,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  2,   4,   1,   1,   1,  8,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  2,   4,   1,   1,   1,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  2,   4,   1,   1,   1,  32,    1,  *fixed_params),

    # DP2 + PP4
    (16,  *gpt_specs["1.3B"],  2,   1,   1,   1,   4,  1,    1,  *fixed_params),
    (32,  *gpt_specs["1.3B"],  2,   1,   1,   1,   4,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   1,   1,   1,   4,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  2,   1,   1,   1,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  2,   1,   1,   1,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  2,   1,   1,   1,   4,  32,    1,  *fixed_params),
    # parax
    (1024,  *gpt_specs["1.3B"],  2,   1,   1,   2,   4,  64,    True,  *fixed_params),

    # DP2 + MP2 + PP2
    (32,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  16,    1,  *fixed_params),
    # parax
    (1024,  *gpt_specs["1.3B"],  2,   2,   1,   4,   2,  32,    False,  *fixed_params),


    # MP-only, max per-gpu batch size = 4
    (32,  *gpt_specs["1.3B"],  1,   8,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  1,   8,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  1,   8,   1,   1,   1, 4,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  1,  8,   1,   1,   1,  8,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  1,  8,   1,   1,   1,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  1,  8,   1,   1,   1,  32,    1,  *fixed_params),

    # MP4 + PP2, max bs = 4
    (32,  *gpt_specs["1.3B"],  1,   4,   1,   1,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  1,   4,   1,   1,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  1,   4,   1,   1,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  1,   4,   1,   1,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  1,   4,   1,   1,   2,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  1,   4,   1,   1,   2,  32,    1,  *fixed_params),

    # MP2 + PP4, max bs = 4
    (32,  *gpt_specs["1.3B"],  1,   2,   1,   1,   4,  1,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  1,   2,   1,   1,   4,  2,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  1,   2,   1,   1,   4,  4,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  1,   2,   1,   1,   4,  8,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  1,   2,   1,   1,   4,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  1,   2,   1,   1,   4,  32,    1,  *fixed_params),

    # PP-only, max per-gpu batch size = 2
    (32,  *gpt_specs["1.3B"],  1,   1,   1,   1,   8,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  1,   1,   1,   1,   8,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"], 1,  1,   1,   1,   8,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"], 1,  1,   1,   1,   8,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"], 1,  1,   1,   1,   8,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"], 1,  1,   1,   1,   8,  64,    True,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["1.3B"], 1,  1,   1,   1,   8,  128,    True,  *fixed_params),

    # ====================
    # 2.7B model
    # Megatron DP maximally can only be 2

    # parax-only, DP = 8 also fails

    # parax-only, DP = 4, MP = 2
    (1024,  *gpt_specs["2.7B"],  4,   2,   1,   4,   1,   32,    False,  *fixed_params),

    # DP 2 + MP 4, max per-gpu bs = 1
    (8,  *gpt_specs["2.7B"],  2,   4,   1,   1,   1,  1,    1,  *fixed_params),
    (16,  *gpt_specs["2.7B"],  2,   4,   1,   1,   1,  2,    1,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  2,   4,   1,   1,   1,  4,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  2,   4,   1,   1,   1,  8,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  2,   4,   1,   1,   1,  16 ,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  2,   4,   1,   1,   1,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  2,   4,   1,   1,   1,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  2,   4,   1,   1,   1,  128,    1,  *fixed_params),

    # DP 2 + MP2 + PP2, max per-gpu bs = 1
    (8,  *gpt_specs["2.7B"],  2,   2,   1,   1,   2,  1,    1,  *fixed_params),
    (16,  *gpt_specs["2.7B"],  2,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  2,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  2,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  2,   2,   1,   1,   2,  16 ,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  2,   2,   1,   1,   2,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  2,   2,   1,   1,   2,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  2,   2,   1,   1,   2,  128,    1,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["2.7B"],  2,   2,   1,   4,   2,  64,    False,  *fixed_params),


    # parax-only DP = 4, PP = 2
    (1024,  *gpt_specs["2.7B"],  4,   1,   1,   4,   2,   64,    True,  *fixed_params),

    # parax-only DP = 2, PP = 4 ?
    (1024,  *gpt_specs["2.7B"],  2,   1,   1,   2,   4,   64,    True,  *fixed_params),

    # MP = 8, max bs = 2
    (16,  *gpt_specs["2.7B"],  1,   8,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  1,   8,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  1,   8,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  1,   8,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  1,   8,   1,   1,   1,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   8,   1,   1,   1,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  1,   8,   1,   1,   1,  128,    1,  *fixed_params),

    # MP = 4, PP = 2, max bs = 2
    (16,  *gpt_specs["2.7B"],  1,   4,   1,   1,   2,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  1,   4,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  1,   4,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  1,   4,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  1,   4,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   4,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  1,   4,   1,   1,   2,  64,    1,  *fixed_params),

    # MP = 2, PP = 4, max bs = 2
    (16,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  64,    1,  *fixed_params),

    # PP = 8, max bs = 1
    (8,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  1,    1,  *fixed_params),
    (16,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  2,    1,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  4,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  8,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  16,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  64,    1,  *fixed_params),
    # parax
    (1024,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  128,    True,  *fixed_params),
],

16: [
    # 2.7B model
    # DP maximally can only be 4 for Megatron

    # parax-only, DP = 16
    (1024, *gpt_specs["2.7B"],  16,   1,   2,   8,   1,  64,    True,  *fixed_params),  # autosharding warning

    # parax-only, DP = 8, PP = 2
    (1024, *gpt_specs["2.7B"],  8,   1,   1,   8,   2,  64,    True,  *fixed_params),
    (1024, *gpt_specs["2.7B"],  8,   1,   1,   8,   2,  32,    True,  *fixed_params),

    # parax-only, DP = 8, MP = 2
    (1024, *gpt_specs["2.7B"],  8,   2,   1,   1,   1,  32,    False,  *fixed_params), # autosharding warning

    # DP = 4, MP =4
    # Megatron-only
    (16,  *gpt_specs["2.7B"],  4,   4,   1,   1,   1,  1,    False,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  4,   4,   1,   1,   1,  2,    False,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  4,   4,   1,   1,   1,  4,    False,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  4,   4,   1,   1,   1,  8,    False,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  4,   4,   1,   1,   1,  16,    False,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  4,   4,   1,   1,   1,  32,    False,  *fixed_params),
    # parax
    (1024,  *gpt_specs["2.7B"],  4,   4,   1,   1,   1,  64,    False,  *fixed_params), # autosharding warning.

    # # DP = 4, MP = 2, PP = 2
    (16,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  1,    False,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  2,    False,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  4,    False,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  8,    False,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  16,    False,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  32,    False,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  64,    False,  *fixed_params),
    # parax
    (1024,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  32,    False,  *fixed_params), # peak memory only 7.5G
    (1024,  *gpt_specs["2.7B"],  4,   2,   1,   8,   2,  64,    False,  *fixed_params),

    # DP = 4, PP = 4
    # impossible even when bs = 1 for megatron
    # parax-only
    (1024,  *gpt_specs["2.7B"],  4,   1,   1,   4,   4,  64,    True,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  4,   1,   1,   4,   4,  32,    True,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  4,   1,   1,   4,   4,  128,    True,  *fixed_params),


    # # DP = 2, MP = 8
    (32,  *gpt_specs["2.7B"],  2,   8,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  2,   8,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  2,   8,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  2,   8,   1,   1,   1,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  2,   8,   1,   1,   1,  32,    1,  *fixed_params),
    # parax
    (1024,  *gpt_specs["2.7B"],  2,   8,   1,   1,   1,  64,    1,  *fixed_params),  # autosharding warning.

    # DP = 2, MP = 4, PP = 2
    (32,  *gpt_specs["2.7B"],  2,   4,   1,   8,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  2,   4,   1,   8,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  2,   4,   1,   8,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  2,   4,   1,   8,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  2,   4,   1,   8,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  2,   4,   1,   8,   2,  64,    1,  *fixed_params),

    # parax-only
    (1024,  *gpt_specs["2.7B"],  2,   4,   1,   8,   2,  64,    False,  *fixed_params), # autosharding warning
    (1024,  *gpt_specs["2.7B"],  2,   4,   1,   8,   2,  32,    False,  *fixed_params), # autosharding warning

    # DP = 2, MP = 2, PP = 4
    (32,  *gpt_specs["2.7B"],  2,   2,   1,   4,   4,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  2,   2,   1,   4,   4,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  2,   2,   1,   4,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  2,   2,   1,   4,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  2,   2,   1,   4,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  2,   2,   1,   4,   4,  64,    1,  *fixed_params),
    # parax-only
    (1024,  *gpt_specs["2.7B"],  2,   2,   1,   4,   4,  64,    False,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  2,   2,   1,   4,   4,  32,    False,  *fixed_params),

    # DP = 2, PP = 8
    (16,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  64,    1,  *fixed_params),
    # parax-only
    (1024,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  64,    True,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  2,   1,   1,   2,   8,  128,    True,  *fixed_params),

    # MP = 8, PP = 2
    (32,  *gpt_specs["2.7B"],  1,   8,   1,   8,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  1,   8,   1,   8,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  1,   8,   1,   8,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  1,   8,   1,   8,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   8,   1,   8,   2,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  1,   8,   1,   8,   2,  32,    1,  *fixed_params),

    # MP = 4, PP = 4
    (32,  *gpt_specs["2.7B"],  1,   4,   1,   4,   4,  1,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  1,   4,   1,   4,   4,  2,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  1,   4,   1,   4,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  1,   4,   1,   4,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   4,   1,   4,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  1,   4,   1,   4,   4,  64,    1,  *fixed_params),

    # MP = 2, PP = 8
    (32,  *gpt_specs["2.7B"],  1,   2,   1,   2,   8,  1,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  1,   2,   1,   2,   8,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  1,   2,   1,   2,   8,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  1,   2,   1,   2,   8,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   2,   1,   2,   8,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  1,   2,   1,   2,   8,  64,    1,  *fixed_params),

    # PP = 16
    (16,  *gpt_specs["2.7B"],  1,   1,   1,   1,   16,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.7B"],  1,   1,   1,   1,   16,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.7B"],  1,   1,   1,   1,   16,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.7B"],  1,   1,   1,   1,   16,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.7B"],  1,   1,   1,   1,   16,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   1,   1,   1,   16,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.7B"],  1,   1,   1,   1,   16,  64,    1,  *fixed_params),

    # parax-only; parax memory OOM at #mb = 64
    (1024,  *gpt_specs["2.7B"],  1,   1,   1,   1,   16,  128,    1,  *fixed_params),


    # ======================================
    # 6.7B model
    # DP maximally can only be 1

    # parax-only, DP = 16:
    (256,  *gpt_specs["6.7B"],  16,   1,   1,   16,   1,  1,    True,  *fixed_params), # autosharding warning.

    # parax-only, DP = 8, PP = 2, Parax OOM.
    # parax-only DP = 4, PP = 4, Parax OOM.
    # parax-only DP = 4, MP = 2, PP = 2, Parax OOM

    # parax-only DP = 4, MP = 4
    # DP = 4, MP = 4:
    (32,  *gpt_specs["6.7B"],  4,   4,   1,   16,   1,  2,    False,  *fixed_params), # cannot run because of bugs

    # parax-only DP = 2, MP = 4, PP = 2
    (1024,  *gpt_specs["6.7B"],  2,   4,   1,   8,   2,  64,    False,  *fixed_params), # autosharding warning

    # parax-only DP = 2, MP = 2, PP = 4
    (1024,  *gpt_specs["6.7B"],  2,   2,   1,   4,   4,  128,    False,  *fixed_params), # autosharding warning.

    # parax-only DP = 2, PP =8
    (1024,  *gpt_specs["6.7B"],  2,   1,   1,   2,   8,  256,    True,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  2,   1,   1,   2,   8,  512,    True,  *fixed_params),


    # MP = 8, PP = 2
    (16,  *gpt_specs["6.7B"],  1,   8,   1,   8,   2,  1,    1,  *fixed_params),
    (32,  *gpt_specs["6.7B"],  1,   8,   1,   8,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["6.7B"],  1,   8,   1,   8,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   8,   1,   8,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   8,   1,   8,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   8,   1,   8,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  1,   8,   1,   8,   2,  64,    1,  *fixed_params),

    # MP = 4, PP = 4
    (8,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  1,    1,  *fixed_params),
    (16,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  2,    1,  *fixed_params),
    (32,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  4,    1,  *fixed_params),
    (64,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  8,    1,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  16,    1,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  32,    1,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  128,    1,  *fixed_params),
    # parax only:
    # (1024,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  128,    False,  *fixed_params), # OOM
    (1024,  *gpt_specs["6.7B"],  1,   4,   1,   4,   4,  256,    False,  *fixed_params), # OOM


    # MP = 2, PP = 8
    (8,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  1,    1,  *fixed_params),
    (16,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  2,    1,  *fixed_params),
    (32,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  4,    1,  *fixed_params),
    (64,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  8,    1,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  16,    1,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  32,    1,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  64,    1,  *fixed_params),
    # parax
    (1024,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  256,    False,  *fixed_params),

    # MP = 1, PP = 16
    (8,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  1,    1,  *fixed_params),
    (16,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  2,    1,  *fixed_params),
    (32,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  4,    1,  *fixed_params),
    (64,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  8,    1,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  32,    1,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  64,    1,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  128,    1,  *fixed_params),
    # parax
    (1024,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  512,    1,  *fixed_params),
],

32: [
    # From now on, we maximize DP if possible.
    #===================
    # 6.7B model, constraints: DP maximally can only be 2, MP maximally can be 8
    # DP = 2, MP = 8, PP = 2
    (32,  *gpt_specs["6.7B"],  2,   8,   2,   8,   2,  1,    False,  *fixed_params),
    (64,  *gpt_specs["6.7B"],  2,   8,   2,   8,   2,  2,    False,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  2,   8,   2,   8,   2,  4,    False,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  2,   8,   2,   8,   2,  8,    False,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  2,   8,   2,   8,   2,  16,    False,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  2,   8,   2,   8,   2,  32,    False,  *fixed_params),

    # DP = 2, MP = 4, PP = 4
    (32,  *gpt_specs["6.7B"],  2,   4,   1,   8,   4,  1,    False,  *fixed_params),
    (64,  *gpt_specs["6.7B"],  2,   4,   1,   8,   4,  2,    False,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  2,   4,   1,   8,   4,  8,    False,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  2,   4,   1,   8,   4,  16,    False,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  2,   4,   1,   8,   4,  32,    False,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  2,   4,   1,   8,   4,  64,    False,  *fixed_params),

    # DP = 2, MP = 2, PP = 8
    (32,  *gpt_specs["6.7B"],  2,   2,   1,   4,   8,  2,    False,  *fixed_params),
    (64,  *gpt_specs["6.7B"],  2,   2,   1,   4,   8,  4,    False,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  2,   2,   1,   4,   8,  8,    False,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  2,   2,   1,   4,   8,  16,    False,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  2,   2,   1,   4,   8,  32,    False,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  2,   2,   1,   4,   8,  64,    False,  *fixed_params),

    # DP = 2， MP = 1, PP = 16
    (64,  *gpt_specs["6.7B"],  2,   1,   1,   2,   16,  8,    True,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  2,   1,   1,   2,   16,  16,    True,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  2,   1,   1,   2,   16,  32,    True,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  2,   1,   1,   2,   16,  64,    True,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  2,   1,   1,   2,   16,  128,    True,  *fixed_params),

    # DP = 1， MP = 8， PP = 4
    (64,  *gpt_specs["6.7B"],  1,   8,   1,   8,   4,  4,    False,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   8,   1,   8,   4,  8,    False,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   8,   1,   8,   4,  16,    False,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   8,   1,   8,   4,  32,    False,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  1,   8,   1,   8,   4,  64,    False,  *fixed_params),

    # DP = 1， MP = 4， PP = 8
    (64,  *gpt_specs["6.7B"],  1,   4,   1,   4,   8,  4,    False,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   4,   1,   4,   8,  8,    False,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   4,   1,   4,   8,  16,    False,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   4,   1,   4,   8,  32,    False,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  1,   4,   1,   4,   8,  64,    False,  *fixed_params),

    # DP = 1， MP = 2， PP = 16
    (64,  *gpt_specs["6.7B"],  1,   2,   1,   2,   16,  4,    False,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   2,   1,   2,   16,  8,    False,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   2,   1,   2,   16,  32,    False,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   2,   1,   2,   16,  64,    False,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  1,   2,   1,   2,   16,  128,    False,  *fixed_params),

    # DP = 1， MP = 1, PP = 32
    (64,  *gpt_specs["6.7B"],  1,   1,   1,   1,   32,  8,    False,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   1,   1,   1,   32,  16,    False,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   1,   1,   1,   32,  64,    False,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   1,   1,   1,   32,  128,    False,  *fixed_params),
    (1024,  *gpt_specs["6.7B"],  1,   1,   1,   1,   32,  256,    False,  *fixed_params),

    #===================
    # 13B model
    # max(MP) = 8, max(DP) = 1， max(PP) = 16
    # MP = 8, PP = 4
    (8,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  1,    False,  *fixed_params),
    (16,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  2,    False,  *fixed_params),
    (32,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  8,    False,  *fixed_params),
    (64,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  16,    False,  *fixed_params),
    (128,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  32,    False,  *fixed_params),
    (256,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  64,    False,  *fixed_params),
    (512,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  128,    False,  *fixed_params),
    (1024,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  256,    False,  *fixed_params),

    # MP = 4, PP = 8
    # max_bs = 4
    (32,  *gpt_specs["15B"],  1,   4,   1,   4,   8,  8,    False,  *fixed_params),
    (64,  *gpt_specs["15B"],  1,   4,   1,   4,   8,  16,    False,  *fixed_params),
    (128,  *gpt_specs["15B"],  1,   4,   1,   4,   8,  32,    False,  *fixed_params),
    (256,  *gpt_specs["15B"],  1,   4,   1,   4,   8,  64,    False,  *fixed_params),
    (512,  *gpt_specs["15B"],  1,   4,   1,   4,   8,  128,    False,  *fixed_params),
    (1024,  *gpt_specs["15B"],  1,   4,   1,   4,   8,  256,    False,  *fixed_params),

    # MP = 2, PP = 16
    (32,  *gpt_specs["15B"],  1,   2,   1,   2,   16,  8,    False,  *fixed_params),
    (64,  *gpt_specs["15B"],  1,   2,   1,   2,   16,  16,    False,  *fixed_params),
    (128,  *gpt_specs["15B"],  1,   2,   1,   2,   16,  32,    False,  *fixed_params),
    (256,  *gpt_specs["15B"],  1,   2,   1,   2,   16,  64,    False,  *fixed_params),
    (512,  *gpt_specs["15B"],  1,   2,   1,   2,   16,  128,    False,  *fixed_params),
    (1024,  *gpt_specs["15B"],  1,   2,   1,   2,   16,  256,    False,  *fixed_params),
],

64: [
    # 13B model

    #==================
    # 39B model

]
}
