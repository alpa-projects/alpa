# suite for gpt benchmarking

gpt_specs = {
# Note: that head_size = hidden_size / #head
        # S，    H，    #L,   #head,  V,
"125M": (1024,  768,   12,    12,   51200, ),
"350M": (1024,  1024,  24,    16,   51200, ),
"760M": (1024,  1536,  24,    16,   51200, ),
"1.3B": (1024,  2048,  24,    24,   51200, ),
"2.7B": (1024,  2560,  32,    32,   51200, ),
"6.7B": (1024,  4096,  32,    32,   51200, ),
"13B":  (1024,  5140,  40,    40,   51200, ),
"39B":  (1024,  8192,  48,    64,   51200, ),
"76B":  (1024,  10240, 60,    80,   51200, ),
}


fixed_params = (True, False, False)
max_global_batch_size = 1024

test_gpt_suite = {
2: [
    # B,       model,           LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, Tie, Auto-layer-slicing
    # (32,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  1,    1,  *fixed_params),
    # (32,  *gpt_specs["350M"],  1,   1,   1,   1,   2,  1,    1,  *fixed_params),
    # (2,  *gpt_specs["760M"],  2,   1,   1,   1,   1,  1,    1,  *fixed_params),

    # (16,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  1,    1,  *fixed_params),
],

4: [
    # (16,  *gpt_specs["760M"],  2,   2,   1,   1,   1,  1,    1,  *fixed_params),
    # (32,  *gpt_specs["760M"],  2,   2,   1,   1,   1,  1,    1,  *fixed_params),
    # (64,  *gpt_specs["760M"],  2,   2,   1,   1,   1,  2,    1,  *fixed_params),
    # (32,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  1,    1,  *fixed_params),

    (32,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  2,    1,  *fixed_params),
]

}


paper_gpt_suite = {

1: [
  # B,       model,           LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, Tie, Auto-layer-slicing
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
    (32,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  4,   1,  *fixed_params),
    (256,  *gpt_specs["350M"],  2,   1,   1,   1,   1,  8,   1,  *fixed_params),

    # MP
    (16,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,   2,   1,   1,   1,  16,    1,  *fixed_params),

    # PP
    (16,  *gpt_specs["350M"],  1,   1,   1,   1,   2,  1,    1,  *fixed_params),
    (32,  *gpt_specs["350M"],  1,   1,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["350M"],  1,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  16,    1,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  64,    1,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  128,    1,  *fixed_params),

    # 760M, cannot train even with bs = 1 per gpu
    # DP all OOM.

    # MP, each 1/2 model can have max batch_size = 8
    (16,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,  2,   1,   1,   1,  8,    1,  *fixed_params),

    # PP, each 1/2 model can have maax batch_size = 8
    (16,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  1,    1,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  2,    1,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  8,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  8,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  16,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  16,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  1,   1,   1,   2,  128,    1,  *fixed_params),
],

4: [
    # 760M, max degree of DP = 2
    (32,  *gpt_specs["760M"],  2,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  2,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["760M"], 2,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (256,  *gpt_specs["760M"], 2,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (512,  *gpt_specs["760M"], 2,   2,   1,   1,   1,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"], 2,   2,   1,   1,   1,  32,    1,  *fixed_params),


    # MP-only, max per-gpu batch size = 8
    (32,  *gpt_specs["760M"],  1,   4,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   4,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,   4,   1,   1,   1, 4,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  8,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  32,    1,  *fixed_params),


    # PP-only
    (32,  *gpt_specs["760M"],  1,   1,   1,   1,   4,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   1,   1,   1,   4,  2,    1,  *fixed_params),
    (128,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  4,    1,  *fixed_params),
    (128,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  64,    1,  *fixed_params),

    # PP + DP
    (32,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  64,    1,  *fixed_params),

    # PP + MP
    # max per-gpu batch = 4
    (32,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  64,    1,  *fixed_params),
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