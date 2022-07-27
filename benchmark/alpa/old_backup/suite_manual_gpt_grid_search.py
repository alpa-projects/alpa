# Grid search on hyperparameters (Deprecated)
"""
              # Remat, RS,   Stage,                 overwrite_global_config_dict
fixed_params = (True,  True, "uniform_layer_gpipe", None)

grid_search_manual = {
    #B,         model,         LD0, LD1, PD0, PD1,  PP,  NB, FM, ...
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

    # alpa-only
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

    #alpa
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  16,    False,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  32,    False,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  64,    False,  *fixed_params),
    (256,  *gpt_specs["350M"],  1,  1,   1,   1,   2,  128,    False,  *fixed_params),

    # 760M, cannot train even with bs = 1 per gpu
    # DP all OOM on megatron

    # alpa-only
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   1,  64,    True,  *fixed_params),
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   1,  32,    True,  *fixed_params),

    # MP, each 1/2 model can have max batch_size = 8
    (16,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   2,   1,   1,   1,  4,    False,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,  2,   1,   1,   1,  8,    False,  *fixed_params),

    # alpa-only
    (512,  *gpt_specs["760M"],  1,  2,   1,   1,   1,  32,    False,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  2,   1,   1,   1,  64,    False,  *fixed_params),

    # PP, each 1/2 model can have maax batch_size = 8
    (16,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  1,    False,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  2,    False,  *fixed_params),
    (32,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  4,    False,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  4,    False,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   1,   1,   1,   2,  8,    False,  *fixed_params),

    # alpa-only
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

    # alpa-only
    (1024,  *gpt_specs["760M"], 4,   1,   1,   1,   1,  16,    True,  *fixed_params),


    # MP-only, max per-gpu batch size = 8, but actullay many cases fail.
    (32,  *gpt_specs["760M"],  1,   4,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   4,   1,   1,   1,  2,    1,  *fixed_params), # OOM
    (128,  *gpt_specs["760M"],  1,   4,   1,   1,   1, 4,    1,  *fixed_params), # OOM
    (256,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  8,    1,  *fixed_params), # OOM
    (512,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  4,   1,   1,   1,  32,    1,  *fixed_params), # OOM

    #alpa-only
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

    # alpa-only
    (1024,  *gpt_specs["760M"], 1,  1,   1,   1,   4,  64,    True,  *fixed_params),

    # PP + DP
    (32,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  2,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  2,  1,   1,   1,   2,  64,    1,  *fixed_params),

    # alpa-only

    (1024,  *gpt_specs["760M"],  2,   1,   1,   2,   2,  32,    True,  *fixed_params),

    # PP + MP
    # max per-gpu batch = 4
    (32,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["760M"],  1,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["760M"],  1,  2,   1,   1,   2,  64,    1,  *fixed_params),

    # alpa-only
    (1024,  *gpt_specs["760M"],  1,   2,   1,   2,   2,  32,    False,  *fixed_params),

    # ====================================== 1.3B model

    # alpa-only
    # Megatron cannot do DP-only
    (1024,  *gpt_specs["1.3B"],  4,   1,   1,   4,   1,  32,    True,  *fixed_params),

    # DP + MP
    (32,  *gpt_specs["1.3B"],  2,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  16,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  32,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"], 2,   2,   1,   1,   1,  128,    1,  *fixed_params),

    # alpa-only
    (1024,  *gpt_specs["1.3B"], 2,   2,   1,   4,   1,  32,    False,  *fixed_params),

    # MP-only, max per-gpu batch size = 4, alpa skips MP-only
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
    # alpa
    (1024,  *gpt_specs["1.3B"], 1,  1,   1,   1,   4,  128,    True,  *fixed_params),


    # PP + DP, max per-gpu batch size = 2
    (32,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  8,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  2,   1,   1,   1,   2,  16,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  32,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  2,  1,   1,   1,   2,  128,    1,  *fixed_params),

    # alpa-only
    (1024,  *gpt_specs["1.3B"],  2,  1,   1,   2,   2,  64,    True,  *fixed_params),

    # PP + MP, alpa skips PP+MP
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

    # DP = 8, Alpa-only
    (1024,  *gpt_specs["1.3B"],  8,   1,   1,   8,   1,   8,    True,  *fixed_params),


    # DP 4 + MP 2, max per-gpu bs = 2
    (16,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  64,    False,  *fixed_params),

    # alpa-only
    (1024,  *gpt_specs["1.3B"],  4,   2,   1,   1,   1,  8,    False,  *fixed_params),

    # DP 4 + PP 2, max per-gpu bs = 2
    (16,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  1,    1,  *fixed_params),
    (32,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  4,   1,   1,   1,   2,  32,    1,  *fixed_params),
    # alpa
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
    # alpa
    (1024,  *gpt_specs["1.3B"],  2,   1,   1,   2,   4,  64,    True,  *fixed_params),

    # DP2 + MP2 + PP2
    (32,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["1.3B"],  2,   2,   1,   1,   2,  16,    1,  *fixed_params),
    # alpa
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

    # alpa-only
    (1024,  *gpt_specs["1.3B"], 1,  1,   1,   1,   8,  128,    True,  *fixed_params),

    # ====================
    # 2.6B model
    # Megatron DP maximally can only be 2

    # alpa-only, DP = 8 also fails

    # alpa-only, DP = 4, MP = 2
    (1024,  *gpt_specs["2.6B"],  4,   2,   1,   4,   1,   32,    False,  *fixed_params),

    # DP 2 + MP 4, max per-gpu bs = 1
    (8,  *gpt_specs["2.6B"],  2,   4,   1,   1,   1,  1,    1,  *fixed_params),
    (16,  *gpt_specs["2.6B"],  2,   4,   1,   1,   1,  2,    1,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  2,   4,   1,   1,   1,  4,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  2,   4,   1,   1,   1,  8,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  2,   4,   1,   1,   1,  16 ,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  2,   4,   1,   1,   1,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  2,   4,   1,   1,   1,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  2,   4,   1,   1,   1,  128,    1,  *fixed_params),

    # DP 2 + MP2 + PP2, max per-gpu bs = 1
    (8,  *gpt_specs["2.6B"],  2,   2,   1,   1,   2,  1,    1,  *fixed_params),
    (16,  *gpt_specs["2.6B"],  2,   2,   1,   1,   2,  2,    1,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  2,   2,   1,   1,   2,  4,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  2,   2,   1,   1,   2,  8,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  2,   2,   1,   1,   2,  16 ,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  2,   2,   1,   1,   2,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  2,   2,   1,   1,   2,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  2,   2,   1,   1,   2,  128,    1,  *fixed_params),

    # alpa-only
    (1024,  *gpt_specs["2.6B"],  2,   2,   1,   4,   2,  64,    False,  *fixed_params),


    # alpa-only DP = 4, PP = 2
    (1024,  *gpt_specs["2.6B"],  4,   1,   1,   4,   2,   64,    True,  *fixed_params),

    # alpa-only DP = 2, PP = 4 ?
    (1024,  *gpt_specs["2.6B"],  2,   1,   1,   2,   4,   64,    True,  *fixed_params),

    # MP = 8, max bs = 2
    (16,  *gpt_specs["2.6B"],  1,   8,   1,   1,   1,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  1,   8,   1,   1,   1,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  1,   8,   1,   1,   1,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  1,   8,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  1,   8,   1,   1,   1,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  1,   8,   1,   1,   1,  64,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  1,   8,   1,   1,   1,  128,    1,  *fixed_params),

    # MP = 4, PP = 2, max bs = 2
    (16,  *gpt_specs["2.6B"],  1,   4,   1,   1,   2,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  1,   4,   1,   1,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  1,   4,   1,   1,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  1,   4,   1,   1,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  1,   4,   1,   1,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  1,   4,   1,   1,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  1,   4,   1,   1,   2,  64,    1,  *fixed_params),

    # MP = 2, PP = 4, max bs = 2
    (16,  *gpt_specs["2.6B"],  1,   2,   1,   1,   4,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  1,   2,   1,   1,   4,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  1,   2,   1,   1,   4,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  1,   2,   1,   1,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  1,   2,   1,   1,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  1,   2,   1,   1,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  1,   2,   1,   1,   4,  64,    1,  *fixed_params),

    # PP = 8, max bs = 1
    (8,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  1,    1,  *fixed_params),
    (16,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  2,    1,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  4,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  8,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  16,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  32,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  64,    1,  *fixed_params),
    # alpa
    (1024,  *gpt_specs["2.6B"],  1,   1,   1,   1,   8,  128,    True,  *fixed_params),
],

16: [
    # 2.6B model
    # DP maximally can only be 4 for Megatron

    # alpa-only, DP = 16
    (1024, *gpt_specs["2.6B"],  16,   1,   2,   8,   1,  64,    True,  *fixed_params),  # autosharding warning

    # alpa-only, DP = 8, PP = 2
    (1024, *gpt_specs["2.6B"],  8,   1,   1,   8,   2,  64,    True,  *fixed_params),
    (1024, *gpt_specs["2.6B"],  8,   1,   1,   8,   2,  32,    True,  *fixed_params),

    # alpa-only, DP = 8, MP = 2
    (1024, *gpt_specs["2.6B"],  8,   2,   1,   1,   1,  32,    False,  *fixed_params), # autosharding warning

    # DP = 4, MP =4
    # Megatron-only
    (16,  *gpt_specs["2.6B"],  4,   4,   1,   1,   1,  1,    False,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  4,   4,   1,   1,   1,  2,    False,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  4,   4,   1,   1,   1,  4,    False,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  4,   4,   1,   1,   1,  8,    False,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  4,   4,   1,   1,   1,  16,    False,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  4,   4,   1,   1,   1,  32,    False,  *fixed_params),
    # alpa
    (1024,  *gpt_specs["2.6B"],  4,   4,   1,   1,   1,  64,    False,  *fixed_params), # autosharding warning.

    # # DP = 4, MP = 2, PP = 2
    (16,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  1,    False,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  2,    False,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  4,    False,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  8,    False,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  16,    False,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  32,    False,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  64,    False,  *fixed_params),
    # alpa
    (1024,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  32,    False,  *fixed_params), # peak memory only 7.5G
    (1024,  *gpt_specs["2.6B"],  4,   2,   1,   8,   2,  64,    False,  *fixed_params),

    # DP = 4, PP = 4
    # impossible even when bs = 1 for megatron
    # alpa-only
    (1024,  *gpt_specs["2.6B"],  4,   1,   1,   4,   4,  64,    True,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  4,   1,   1,   4,   4,  32,    True,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  4,   1,   1,   4,   4,  128,    True,  *fixed_params),


    # # DP = 2, MP = 8
    (32,  *gpt_specs["2.6B"],  2,   8,   1,   1,   1,  1,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  2,   8,   1,   1,   1,  2,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  2,   8,   1,   1,   1,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  2,   8,   1,   1,   1,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  2,   8,   1,   1,   1,  32,    1,  *fixed_params),
    # alpa
    (1024,  *gpt_specs["2.6B"],  2,   8,   1,   1,   1,  64,    1,  *fixed_params),  # autosharding warning.

    # DP = 2, MP = 4, PP = 2
    (32,  *gpt_specs["2.6B"],  2,   4,   1,   8,   2,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  2,   4,   1,   8,   2,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  2,   4,   1,   8,   2,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  2,   4,   1,   8,   2,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  2,   4,   1,   8,   2,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  2,   4,   1,   8,   2,  64,    1,  *fixed_params),

    # alpa-only
    (1024,  *gpt_specs["2.6B"],  2,   4,   1,   8,   2,  64,    False,  *fixed_params), # autosharding warning
    (1024,  *gpt_specs["2.6B"],  2,   4,   1,   8,   2,  32,    False,  *fixed_params), # autosharding warning

    # DP = 2, MP = 2, PP = 4
    (32,  *gpt_specs["2.6B"],  2,   2,   1,   4,   4,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  2,   2,   1,   4,   4,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  2,   2,   1,   4,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  2,   2,   1,   4,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  2,   2,   1,   4,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  2,   2,   1,   4,   4,  64,    1,  *fixed_params),
    # alpa-only
    (1024,  *gpt_specs["2.6B"],  2,   2,   1,   4,   4,  64,    False,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  2,   2,   1,   4,   4,  32,    False,  *fixed_params),

    # DP = 2, PP = 8
    (16,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  64,    1,  *fixed_params),
    # alpa-only
    (1024,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  64,    True,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  2,   1,   1,   2,   8,  128,    True,  *fixed_params),

    # MP = 8, PP = 2
    (32,  *gpt_specs["2.6B"],  1,   8,   1,   8,   2,  1,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  1,   8,   1,   8,   2,  2,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  1,   8,   1,   8,   2,  4,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  1,   8,   1,   8,   2,  8,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  1,   8,   1,   8,   2,  16,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  1,   8,   1,   8,   2,  32,    1,  *fixed_params),

    # MP = 4, PP = 4
    (32,  *gpt_specs["2.6B"],  1,   4,   1,   4,   4,  1,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  1,   4,   1,   4,   4,  2,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  1,   4,   1,   4,   4,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  1,   4,   1,   4,   4,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  1,   4,   1,   4,   4,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  1,   4,   1,   4,   4,  64,    1,  *fixed_params),

    # MP = 2, PP = 8
    (32,  *gpt_specs["2.6B"],  1,   2,   1,   2,   8,  1,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  1,   2,   1,   2,   8,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  1,   2,   1,   2,   8,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  1,   2,   1,   2,   8,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  1,   2,   1,   2,   8,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  1,   2,   1,   2,   8,  64,    1,  *fixed_params),

    # PP = 16
    (16,  *gpt_specs["2.6B"],  1,   1,   1,   1,   16,  1,    1,  *fixed_params),
    (32,  *gpt_specs["2.6B"],  1,   1,   1,   1,   16,  2,    1,  *fixed_params),
    (64,  *gpt_specs["2.6B"],  1,   1,   1,   1,   16,  4,    1,  *fixed_params),
    (128,  *gpt_specs["2.6B"],  1,   1,   1,   1,   16,  8,    1,  *fixed_params),
    (256,  *gpt_specs["2.6B"],  1,   1,   1,   1,   16,  16,    1,  *fixed_params),
    (512,  *gpt_specs["2.6B"],  1,   1,   1,   1,   16,  32,    1,  *fixed_params),
    (1024,  *gpt_specs["2.6B"],  1,   1,   1,   1,   16,  64,    1,  *fixed_params),

    # alpa-only; alpa memory OOM at #mb = 64
    (1024,  *gpt_specs["2.6B"],  1,   1,   1,   1,   16,  128,    1,  *fixed_params),


    # ======================================
    # 6.7B model
    # DP maximally can only be 1

    # alpa-only, DP = 16:
    (256,  *gpt_specs["6.7B"],  16,   1,   1,   16,   1,  1,    True,  *fixed_params), # autosharding warning.

    # alpa-only, DP = 8, PP = 2, Alpa OOM.
    # alpa-only DP = 4, PP = 4, Alpa OOM.
    # alpa-only DP = 4, MP = 2, PP = 2, Alpa OOM

    # alpa-only DP = 4, MP = 4
    # DP = 4, MP = 4:
    (32,  *gpt_specs["6.7B"],  4,   4,   1,   16,   1,  2,    False,  *fixed_params), # cannot run because of bugs

    # alpa-only DP = 2, MP = 4, PP = 2
    (1024,  *gpt_specs["6.7B"],  2,   4,   1,   8,   2,  64,    False,  *fixed_params), # autosharding warning

    # alpa-only DP = 2, MP = 2, PP = 4
    (1024,  *gpt_specs["6.7B"],  2,   2,   1,   4,   4,  128,    False,  *fixed_params), # autosharding warning.

    # alpa-only DP = 2, PP =8
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
    # alpa only:
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
    # alpa
    (1024,  *gpt_specs["6.7B"],  1,   2,   1,   2,   8,  256,    False,  *fixed_params),

    # MP = 1, PP = 16
    (8,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  1,    1,  *fixed_params),
    (16,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  2,    1,  *fixed_params),
    (32,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  4,    1,  *fixed_params),
    (64,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  8,    1,  *fixed_params),
    (128,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  32,    1,  *fixed_params),
    (256,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  64,    1,  *fixed_params),
    (512,  *gpt_specs["6.7B"],  1,   1,   1,   1,   16,  128,    1,  *fixed_params),
    # alpa
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

    # ==================
    # Alpa suites
    # ==================
    # 6.7B model
    # DP = 32 does not work
    # DP = 16, MP = 2 does not work
    # DP = 16, PP = 2 does not work
    #
    # DP = 8, MP = 4
    (1024,  *gpt_specs["6.7B"],  8,   4,   1,   1,   1,  64,    False,  *fixed_params),
    # DP = 4, MP = 8
    (1024,  *gpt_specs["6.7B"],  4,   8,   1,   1,   1,  16,    False,  *fixed_params),

    # PP at least = 4, otherwise MP or DP will cross noe.
    # DP = 8, PP = 4
    (1024,  *gpt_specs["6.7B"],  8,   1,   1,   8,   4,  128,    True,  *fixed_params),

    # DP = 8, MP = 2, PP = 2, I think this will be bad because DP is crossing nodes.
    (1024,  *gpt_specs["6.7B"],  8,   2,   2,   8,   2,  128,    False,  *fixed_params),

    # DP = 4, PP = 8
    (1024,  *gpt_specs["6.7B"],  4,   1,   1,   4,   8,  64,    True,  *fixed_params),

    # dp = 4, mp = 2, pp = 4
    (1024,  *gpt_specs["6.7B"],  4,   2,   1,   8,   4,  32,    False,  *fixed_params),

    # DP = 2, PP = 16
    (1024,  *gpt_specs["6.7B"],  2,   1,   1,   2,   16,  128,    True,  *fixed_params),

    # DP = 2, MP = 2, PP = 8
    (1024,  *gpt_specs["6.7B"],  2,   2,   1,   4,   8,  64,    False,  *fixed_params),

    # DP = 2, MP = 4, PP = 4, bad
    # MP = 8, PP = 4, bad
    # MP = 4, PP = 8, bad
    # MP = 2, PP = 16, bad

    # PP = 32
    # microbatch_size = 32 does not work.
    (1024,  *gpt_specs["6.7B"],  1,   1,   1,   1,   32,  512,    True,  *fixed_params),

    #===================
    # 15B model
    # DP = 32 does not work
    # DP = 16, MP = 2 does not work
    # DP = 16, PP = 2 does not work
    # DP = 8, MP = 4 impossible
    # DP = 8, PP = 4, impossible
    # DP = 8, MP = 2, PP = 2, impossible
    # DP = 4, MP = 8, bad because DP cross nodes
    # DP = 4, PP = 8, impossible

    # DP = 4, mp = 4, pp = 2
    (64,  *gpt_specs["15B"],  4,   4,   2,   8,   2,  2,    True,  *fixed_params), # buggy

    # dp = 4, mp = 2, pp = 4, impoosible
    # DP = 2, PP = 16, impossible

    # DP = 2, MP = 2, PP = 8
    (64,  *gpt_specs["15B"],  2,   2,   1,   4,   8,  2,    False,  *fixed_params),

    # DP = 2, MP = 4, PP = 4
    (1024,  *gpt_specs["15B"],  2,   4,   1,   8,   4,  256,    False,  *fixed_params),

    # DP = 2, MP = 8, PP = 2, bad
    (64,  *gpt_specs["15B"],  2,   8,   2,   8,   2,  2,    False,  *fixed_params),

    # MP = 8, PP = 4
    (1024,  *gpt_specs["15B"],  1,   8,   1,   8,   4,  256,    False,  *fixed_params),

    # MP = 4, PP = 8
    (8,  *gpt_specs["15B"],  1,   4,   1,   4,   8,   2,    False,  *fixed_params),

    # MP = 2, PP = 16
    (1024,  *gpt_specs["15B"],  1,   2,   1,   2,   16,  512,    False,  *fixed_params),

    # PP = 32, impossible, needs either 32 or 64 layers
],

64: [
    # 39B model
    (1024,  *gpt_specs["39B"],  1,   4,   1,   4,   16,  1024,    True,  *fixed_params),
    (1024,  *gpt_specs["39B"],  1,   8,   1,   8,   8,  512,    True,  *fixed_params),
    # Alpa suites

]
}
"""
