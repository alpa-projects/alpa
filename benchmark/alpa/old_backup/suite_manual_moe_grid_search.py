"""
# Grid search on hyperparameters (deprecated)
#               Remat, RS,   Stage
fixed_params = (True,  True, "uniform_layer_gpipe")

grid_search_manual = {
1: [
    # B,      model,  LD0,  LD1,  PD0,  PD1,  PP,  NB,   FM,   (Remat, RS, Stage), _
    #1 GPUs, deepspeed max bs = 8, alpa = 16
    (8,     *moe_specs["380M"],  8 * 1024 // 2,   1,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (32,    *moe_specs["380M"],  8 * 1024 // 2,   1,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (128,    *moe_specs["380M"], 8 * 1024 // 2,   1,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    (512,    *moe_specs["380M"], 8 * 1024 // 2,   1,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),
    (1024,    *moe_specs["380M"], 1024 * 2,   1,    1,    1,    1,   1,    128,  True,   *fixed_params,     1),

    # ==================
    # Alpa suites
    # ==================
    (1024,    *moe_specs["380M"], 1024 * 2,   1,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),
],

2: [
    # 2 GPUs, deepspeed max effective bs = 8
    # MP is always worse
    # ===============================
    # 380M model
    # B,      model,                              LD0,  LD1,  PD0,  PD1,  PP,  NB,   FM,   (Remat, RS, Stage), _
    (16,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (64,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (256,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    (1024,     *moe_specs["380M"],  1024 * 2,   2,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # DP = 2, MP = 2, deepspeed will borrow 2 DP to MP at MOE layers.
    (16,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (64,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (256,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     2),
    (1024,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    64,  True,   *fixed_params,     2),

    # ================================
    # 690M model
    # DP = 2, EP = 1
    (16,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    (64,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    4,  True,   *fixed_params,     1),
    (256,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    (1024,     *moe_specs["690M"],  1024 * 2,   2,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # DP = 2, EP = 2
    (16,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (64,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (256,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     2),
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    64,  True,   *fixed_params,     2),

    # DP = 1, MP = 2, EP = 1
    (1024,     *moe_specs["690M"],  8 * 1024 // 2,   1,    2,    1,    1,   1,    128,  True,   *fixed_params,     1),

    # ==================
    # Alpa suites
    # ==================
    # alpa-only,
    (1024,     *moe_specs["380M"],  1024 * 2,   2,    1,    1,    1,   1,    32,  True,   *fixed_params,     1),
    (1024,     *moe_specs["380M"],  1024 * 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),

    # alpa-only, DP = 1, MP = 2
    (1024,     *moe_specs["380M"],  1024 * 2,   1,    2,    1,    1,   1,    64,  False,   *fixed_params,     2),
    (1024,     *moe_specs["380M"],  1024 * 2,   1,    2,    1,    1,   1,    16,  False,   *fixed_params,     2),

    # alpa-only, PP = 2
    (1024,     *moe_specs["380M"],  1024 * 2,   1,    1,    1,    1,   2,    64,  True,   *fixed_params,     2),

    # ================================
    # 690M model
    # alpa-only, DP = 2
    (1024,     *moe_specs["690M"],  1024 * 2,   2,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),

    # alpa-only, DP = 1, MP = 2
    (1024,     *moe_specs["690M"],  1024 * 2,   1,    2,    1,    1,   1,    64,  False,   *fixed_params,     2),
    (1024,     *moe_specs["690M"],  1024 * 2,   1,    2,    1,    1,   1,    32,  False,   *fixed_params,     2),
    (1024,     *moe_specs["690M"],  1024 * 2,   1,    2,    1,    1,   1,    32,  False,   *fixed_params,     2),

    # alpa-only, PP = 2
    (1024,     *moe_specs["690M"],  1024 * 2,   1,    1,    1,    1,   2,    64,  True,   *fixed_params,     2),
],

4: [
    # B,      model,                              LD0,  LD1,  PD0,  PD1,  PP,  NB,   FM,   (Remat, RS, Stage), _
    (16,     *moe_specs["380M"],  8 * 1024 // 2,   2,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
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


    # ==================
    # Alpa suites
    # ==================
    # 690M model
    # DP = 4
    (1024,     *moe_specs["690M"],  1024 * 2,   4,    1,    1,    1,   1,    8,  True,   *fixed_params,     1),

    # DP = 2, MP = 2,
    (1024,     *moe_specs["690M"],  1024 * 2,   2,    2,    1,    1,   1,    8,  False,   *fixed_params,     1),
    # skipping: DP = 1, MP = 4

    # DP = 2, PP = 2
    (1024,     *moe_specs["690M"],  1024 * 2,   2,    1,    1,    2,   2,    32,  True,   *fixed_params,     1),

    # PP = 4
    (1024,     *moe_specs["690M"],  1024 * 2,   1,    1,    1,    1,   4,    64,  False,   *fixed_params,     1),

    # MP = 2, PP = 2
    (1024,     *moe_specs["690M"],  1024 * 2,   1,    2,    1,    2,   2,    32,  False,   *fixed_params,     1),


    # ================================
    # 1.3B model
    # DP = 4
    (1024,     *moe_specs["1.3B"],  1024 * 2,   4,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"],  1024 * 2,   4,    1,    1,    1,   1,    32,  True,   *fixed_params,     1),

    # DP = 2, MP = 2
    # (1024,     *moe_specs["1.3B"],  1024 * 2,   2,    2,    1,    1,   1,    16,  True,   *fixed_params,     1),  # warning
    #
    # DP = 2, PP = 2
    (1024,     *moe_specs["1.3B"],  1024 * 2,   2,    1,    1,    2,   2,    64,  True,   *fixed_params,     1),
    # PP = 4
    (1024,     *moe_specs["1.3B"],  1024 * 2,   1,    1,    1,    1,   4,    64,  True,   *fixed_params,     1),
    # MP = 2, PP = 2
    (1024,     *moe_specs["1.3B"],  1024 * 2,   1,    2,    1,    2,   2,    32,  True,   *fixed_params,     1),
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

    # ==================
    # Alpa suites
    # ==================
    # 1.3B model
    # DP = 8
    (1024,     *moe_specs["1.3B"], 1024 * 1024,   8,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"], 1024 * 1024,   8,    1,    1,    1,   1,    8,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"], 1024 * 2,   8,    1,    1,    1,   1,    16,  False,   *fixed_params,     1),

    # DP = 2, MP = 4
    (1024,     *moe_specs["1.3B"], 1024 * 2,   2,    4,    1,    1,   1,    8,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"], 1024 * 2,   2,    4,    1,    1,   1,    8,  False,   *fixed_params,     1),

    # DP = 4, MP = 2
    (1024,     *moe_specs["1.3B"], 1024 * 2,   4,    2,    1,    1,   1,    8,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"], 1024 * 2,   4,    2,    1,    1,   1,    8,  False,   *fixed_params,     1),

    # DP = 1, MP = 8
    (1024,     *moe_specs["1.3B"], 1024 * 2,   1,    8,    1,    1,   1,    32,  True,   *fixed_params,     1),

    # DP = 4, PP = 2
    (1024,     *moe_specs["1.3B"], 1024 * 2,   4,    1,    1,    4,   2,    16,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"], 1024 * 1024,   4,    1,    1,    4,   2,    16,  False,   *fixed_params,     1),


    # DP = 2, PP = 4
    (1024,     *moe_specs["1.3B"], 1024 * 2,   2,    1,    1,    2,   4,    32,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"],  1024 * 1024,   2,    1,    1,    2,   4,    32,  False,   *fixed_params,     1),

    # DP = 2, MP = 2, PP = 2
    (1024,     *moe_specs["1.3B"], 1024 * 2,   2,    2,    1,    4,   2,    16,  True,   *fixed_params,     1),
    (1024,     *moe_specs["1.3B"],  1024 * 1024,   2,    2,    1,    4,   2,    32,  False,   *fixed_params,     1),

    # MP = 4, PP = 2
    (1024,     *moe_specs["1.3B"], 1024 * 2,   1,    4,    1,    4,   2,    16,  True,   *fixed_params,     1),

    # MP = 2, PP = 4
    (1024,     *moe_specs["1.3B"], 1024 * 2,   1,    2,    1,    2,   4,    16,  True,   *fixed_params,     1),

    # PP = 8
    (1024,     *moe_specs["1.3B"], 1024 * 2,   1,    1,    1,    1,   8,    16,  True,   *fixed_params,     1),


    # ================================
    # 2.4B model
    # DP = 8, impossible
    (1024,     *moe_specs["2.4B"],  1024 * 2,   8,    1,    1,    1,   1,    64,  False,   *fixed_params,     1),
    (1024,     *moe_specs["2.4B"],  1024 * 2,   8,    1,    1,    1,   1,    16,  False,   *fixed_params,     1),

    (1024,     *moe_specs["2.4B"],  1024 * 2,   4,    2,    1,    1,   1,    64,  False,   *fixed_params,     1),

    (1024,     *moe_specs["2.4B"],  1024 * 2,   2,    4,    1,    1,   1,    64,  False,   *fixed_params,     1),

    # DP = 4, MP = 2
    # DP = 2, MP = 4
    # DP = 1, MP = 8
    # DP = 4, PP = 2
    (1024,     *moe_specs["2.4B"],  1024 * 1024,   4,    1,    1,    4,   2,    32,  False,  *fixed_params,    1),
    # DP = 2, PP = 4
    (1024,     *moe_specs["2.4B"],  1024 * 1024,   2,    1,    1,    2,   4,    32,  False,   *fixed_params,     1),
    # DP = 2, MP = 2, PP = 2
    (1024,     *moe_specs["2.4B"],  1024 * 1024,   2,    2,    1,    4,   2,    32,  False,   *fixed_params,     1),
    # MP = 4, PP = 2
    # PP = 8
    (1024,     *moe_specs["2.4B"],  1024 * 1024,   1,    1,    1,    1,   8,    64,  False,   *fixed_params,     1),
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
    (128,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    2,  True,   *fixed_params,     2),
    (512,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     2),
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    16,  True,   *fixed_params,     2),

    # DP = 16, EP = 4
    (128,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (512,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     4),

    # DP = 16， EP = 8
    (128,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (512,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     8),

    # DP = 16， EP = 16
    (128,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    1,  True,   *fixed_params,     16),
    (512,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    4,  True,   *fixed_params,     16),
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   16,    1,    1,    1,   1,    8,  True,   *fixed_params,     16),

    # DP = 8， MP = 2， EP = 4
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   8,    2,    1,    1,   1,    16,  True,   *fixed_params,     4),

    # DP = 4， MP = 4， EP = 4
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   4,    4,    1,    1,   1,    32,  True,   *fixed_params,     4),


    # ==================
    # Alpa suites
    # ==================

],

32: [
    # ================================
    # 4.5B model, max(ep) = 8, otherwise extremely slow.
    # DP = 32, EP = 1, cannot accumulate..
    (32,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     1),
    # (256,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    16,  True,   *fixed_params,     1),
    # (1024,     *moe_specs["4.5B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    64,  True,   *fixed_params,     1),

    # DP = 32, EP = 2
    (256,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     2),
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     2),
    (2048,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     2),

    # DP = 32, EP = 4
    (256,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     4),
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     4),
    (2048,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     4),

    # DP = 32, EP = 8
    (256,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (2048,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     8),

    # DP = 32, EP = 16
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     16),

    # DP = 32, EP = 32
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     32),

    # EP * MP <= 8
    # DP = 8， MP = 4， EP = 2
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   8,    4,    1,    1,   1,    16,  True,   *fixed_params,     2),

    # DP = 8， MP = 2， EP = 4
    (1024,     *moe_specs["10B"],  8 * 1024 // 2,   16,    2,    1,    1,   1,    8,  True,   *fixed_params,     4),

    # ================================
    # 27B model
    # DP = 32, EP = 1 fails when batch size = 32
    # DP = 32, EP = 2 fails when batch size = 32
    # DP = 32 , EP = 4 fails when batch size = 32
    # DP = 32, EP = 8
    (128,     *moe_specs["27B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    1,  True,   *fixed_params,     8),
    (512,     *moe_specs["27B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    4,  True,   *fixed_params,     8),
    (1024,     *moe_specs["27B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     8),

    # DP = 32, EP = 16
    (1024,     *moe_specs["27B"],  8 * 1024 // 2,   32,    1,    1,    1,   1,    8,  True,   *fixed_params,     16),

    # EP * MP == 8
    # DP = 16, MP = 2, EP = 4
    (1024,     *moe_specs["27B"],  8 * 1024 // 2,   16,    2,    1,    1,   1,    16,  True,   *fixed_params,     4),


    # I skip the two below
    # DP = 8, MP = 4, EP = 2
    (1024,     *moe_specs["27B"],  8 * 1024 // 2,   8,    4,    1,    1,   1,    16,  True,   *fixed_params,     2),

    # DP = 4, MP = 8, EP = 1
    (1024,     *moe_specs["27B"],  8 * 1024 // 2,   4,    8,    1,    1,   1,    8,  True,   *fixed_params,     1),


    # ==================
    # Alpa suites
    # ==================
    # 4.5B+ model
    (1024,     *moe_specs["10B"],  1024 * 2,   8,    1,    1,    8,   4,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["10B"],  1024 * 2,   4,    2,    1,    8,   4,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["10B"],  1024 * 2,   4,    1,    1,    4,   8,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["10B"],  1024 * 2,   2,    1,    1,    2,   16,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["10B"],  1024 * 2,   1,    1,    1,    1,   32,   128,  False,   *fixed_params,     1),
    (1024,     *moe_specs["10B"],  1024 * 2,   2,    2,    1,    4,   8,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["10B"],  1024 * 2,   2,    4,    1,    8,   4,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["10B"],  1024 * 2,   1,    8,    1,    8,   4,    32,  True,   *fixed_params,     1),

    # 27B model
    (1024,     *moe_specs["27B"],  1024 * 2,   8,    1,    1,    8,   4,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["27B"],  1024 * 2,   4,    2,    1,    8,   4,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["27B"],  1024 * 2,   4,    1,    1,    4,   8,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["27B"],  1024 * 2,   2,    1,    1,    2,   16,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["27B"],  1024 * 2,   1,    1,    1,    1,   32,   256,  False,   *fixed_params,     1),
    (1024,     *moe_specs["27B"],  1024 * 2,   2,    2,    1,    4,   8,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["27B"],  1024 * 2,   2,    4,    1,    8,   4,    32,  False,   *fixed_params,     1),
    (1024,     *moe_specs["27B"],  1024 * 2,   1,    8,    1,    8,   4,    32,  True,   *fixed_params,     1),
],

64: [
]
}
"""
