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
"13B":  (1024,  5140,  40,    40,   51200, ),
"39B":  (1024,  8192,  48,    64,   51200, ),
"76B":  (1024,  10240, 60,    80,   51200, ),
}


fixed_params = (True, False, False)
max_global_batch_size = 1024

test_gpt_suite = {
8: [
    # (32,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  1,    1,  *fixed_params),
    # (32,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  2,    1,  *fixed_params),
    # (32,  *gpt_specs["1.3B"],  2,   4,   1,   1,   1,  1,    1,  *fixed_params),
    # (64,  *gpt_specs["1.3B"],  2,   4,   1,   1,   1,  1,    1,  *fixed_params),

    # (16,  *gpt_specs["1.3B"],  2,   1,   1,   1,   4,  1,    1,  *fixed_params),
    #
    # (32,  *gpt_specs["1.3B"],  1,   1,   1,   1,   8,  2,    1,  *fixed_params),
    # (16,  *gpt_specs["2.7B"],  4,   2,   1,   1,   1,  1,    1,  *fixed_params),


    (256,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   2,   1,   1,   4,  32,    1,  *fixed_params),

    (256,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  64,    1,  *fixed_params),
],

16: [
    # 2.7B model


    # 6.7B model
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


    # MP-only, max per-gpu batch size = 8, but actullay many cases fail.
    (32,  *gpt_specs["760M"],  1,   4,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   4,   1,   1,   1,  2,    1,  *fixed_params), # OOM
    (128,  *gpt_specs["760M"],  1,   4,   1,   1,   1, 4,    1,  *fixed_params), # OOM
    (256,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  8,    1,  *fixed_params), # OOM
    (512,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  32,    1,  *fixed_params), # OOM


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


    # 1.3B model
    # DP + MP
    (32,  *gpt_specs["1.3B"],  2,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  16,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  32,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  128,    1,  *fixed_params),

    # MP-only, max per-gpu batch size = 4
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
    (1024,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  128,    1,  *fixed_params),

    # PP + DP, max per-gpu batch size = 2
    (32,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  8,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  16,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  128,    1,  *fixed_params),

    # PP + MP
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
    # DP 4 + MP 2, max per-gpu bs = 2
    (16,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  64,    1,  *fixed_params),

    # DP 4 + PP 2, max per-gpu bs = 2
    (16,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  1,    1,  *fixed_params),
    (32,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  64,    1,  *fixed_params),

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
    (1024,  *gpt_specs["1.3B"],  2,   1,   1,   1,   4,  64,    1,  *fixed_params),

    # DP2 + MP2 + PP2
    (32,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  32,    1,  *fixed_params),

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
    (64,  *gpt_specs["1.3B"],  1,   1,   1,   1,   4,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  64,    1,  *fixed_params),

    # ====================
    # 2.7B model
    # DP maximally can only be 2

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
    (1024,  *gpt_specs["2.7B"],  1,   1,   1,   1,   8,  128,    1,  *fixed_params),
],

16: [
    # 2.7B model
    # DP maximally can only be 4
    # DP = 4, MP =4
    (8,  *gpt_specs["2.7B"],  4,   4,   1,   1,   1,  1,    1,  *fixed_params),

    # DP = 4, MP = 2, PP = 2
    (8,  *gpt_specs["2.7B"],  4,   2,   1,   1,   2,  1,    1,  *fixed_params),

    # DP = 4, PP = 4
    (8,  *gpt_specs["2.7B"],  4,   1,   1,   1,   4,  1,    1,  *fixed_params),

    # DP = 2, MP = 8
    (8,  *gpt_specs["2.7B"],  2,   8,   1,   1,   4,  1,    1,  *fixed_params),

    # DP = 2, MP = 4, PP = 2

    # DP = 2, MP = 2, PP = 4

    # DP = 2, PP = 8

    # MP = 8, PP = 2

    # MP = 4, PP = 4

    # MP = 2, PP = 8

    # MP = 1, PP = 16

    # =================
    # 6.7B model
    # TODO (figure out the largest available microBS)


],

32: [
    # From now on, we maximize DP if possible.
    # 6.7 B model

    #==================
    # 13B model
],

64: [
    # 13B model

    #==================
    # 39B model

]
}