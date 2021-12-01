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
}

dummy_arguments = (0, 0, 0, 0, 0) # LD0, LD1, PD0, PD1, PP, not used for auto
fixed_params = (False, True, True, True) # FD,  Remat, RS, Auto layer & stage
max_global_batch_size = 1024

paper_auto_gpt_suite = {
    # B,       model,         dummy,            NB(#microbatches)
1: [
    # 125M
    (2,    *gpt_specs["125M"], *dummy_arguments, 1, *fixed_params),
    (4,    *gpt_specs["125M"], *dummy_arguments, 1, *fixed_params),
    (8,    *gpt_specs["125M"], *dummy_arguments, 1, *fixed_params),
    (16,   *gpt_specs["125M"], *dummy_arguments, 1, *fixed_params),
    (32,   *gpt_specs["125M"], *dummy_arguments, 1, *fixed_params),
    (64,   *gpt_specs["125M"], *dummy_arguments, 2, *fixed_params),
    (128,  *gpt_specs["125M"], *dummy_arguments, 4, *fixed_params),

    # 350M
    (2,    *gpt_specs["350M"], *dummy_arguments, 1, *fixed_params),
    (4,    *gpt_specs["350M"], *dummy_arguments, 1, *fixed_params),
    (8,    *gpt_specs["350M"], *dummy_arguments, 1, *fixed_params),
    (16,   *gpt_specs["350M"], *dummy_arguments, 1, *fixed_params),
    (32,   *gpt_specs["350M"], *dummy_arguments, 2, *fixed_params),
    (64,   *gpt_specs["350M"], *dummy_arguments, 4, *fixed_params),
    (128,  *gpt_specs["350M"], *dummy_arguments, 8, *fixed_params),
],

2: [
    # 350M, max_bs = 8 per gpu (whole model)
    (16,  *gpt_specs["350M"], *dummy_arguments, 1,   *fixed_params),
    (32,  *gpt_specs["350M"], *dummy_arguments, 1,   *fixed_params),
    (32,  *gpt_specs["350M"], *dummy_arguments, 2,   *fixed_params),
    (64,  *gpt_specs["350M"], *dummy_arguments, 2,   *fixed_params),
    (64,  *gpt_specs["350M"], *dummy_arguments, 4,   *fixed_params),
    (128, *gpt_specs["350M"], *dummy_arguments, 4,   *fixed_params),
    (128, *gpt_specs["350M"], *dummy_arguments, 8,   *fixed_params),
    (256, *gpt_specs["350M"], *dummy_arguments, 4,   *fixed_params),
    (256, *gpt_specs["350M"], *dummy_arguments, 8,   *fixed_params),
    (256, *gpt_specs["350M"], *dummy_arguments, 16,  *fixed_params),
    (256, *gpt_specs["350M"], *dummy_arguments, 32,  *fixed_params),
    (256, *gpt_specs["350M"], *dummy_arguments, 64,  *fixed_params),
    (256, *gpt_specs["350M"], *dummy_arguments, 128, *fixed_params),


    # 760M, cannot train even with bs = 1 per gpu
    (16,   *gpt_specs["760M"], *dummy_arguments, 1,   *fixed_params),
    (32,   *gpt_specs["760M"], *dummy_arguments, 2,   *fixed_params),
    (32,   *gpt_specs["760M"], *dummy_arguments, 4,   *fixed_params),
    (64,   *gpt_specs["760M"], *dummy_arguments, 4,   *fixed_params),
    (64,   *gpt_specs["760M"], *dummy_arguments, 8,   *fixed_params),
    (128,  *gpt_specs["760M"], *dummy_arguments, 8,   *fixed_params),
    (128,  *gpt_specs["760M"], *dummy_arguments, 16,  *fixed_params),
    (256,  *gpt_specs["760M"], *dummy_arguments, 16,  *fixed_params),
    (256,  *gpt_specs["760M"], *dummy_arguments, 32,  *fixed_params),
    (512,  *gpt_specs["760M"], *dummy_arguments, 32,  *fixed_params),
    (512,  *gpt_specs["760M"], *dummy_arguments, 64,  *fixed_params),
    (1024, *gpt_specs["760M"], *dummy_arguments, 32,  *fixed_params),
    (1024, *gpt_specs["760M"], *dummy_arguments, 64,  *fixed_params),
    (1024, *gpt_specs["760M"], *dummy_arguments, 128, *fixed_params),
],

4: [
    (32,   *gpt_specs["760M"], *dummy_arguments, 1,  *fixed_params),
    (32,   *gpt_specs["760M"], *dummy_arguments, 2,  *fixed_params),
    (64,   *gpt_specs["760M"], *dummy_arguments, 2,  *fixed_params),
    (64,   *gpt_specs["760M"], *dummy_arguments, 4,  *fixed_params),
    (128,  *gpt_specs["760M"], *dummy_arguments, 4,  *fixed_params),
    (128,  *gpt_specs["760M"], *dummy_arguments, 8,  *fixed_params),
    (256,  *gpt_specs["760M"], *dummy_arguments, 8,  *fixed_params),
    (256,  *gpt_specs["760M"], *dummy_arguments, 16, *fixed_params),
    (512,  *gpt_specs["760M"], *dummy_arguments, 16, *fixed_params),
    (512,  *gpt_specs["760M"], *dummy_arguments, 32, *fixed_params),
    (1024, *gpt_specs["760M"], *dummy_arguments, 16, *fixed_params),
    (1024, *gpt_specs["760M"], *dummy_arguments, 32, *fixed_params),
    (1024, *gpt_specs["760M"], *dummy_arguments, 64, *fixed_params),

    (32,   *gpt_specs["1.3B"], *dummy_arguments, 2,   *fixed_params),
    (32,   *gpt_specs["1.3B"], *dummy_arguments, 4,   *fixed_params),
    (64,   *gpt_specs["1.3B"], *dummy_arguments, 4,   *fixed_params),
    (64,   *gpt_specs["1.3B"], *dummy_arguments, 8,   *fixed_params),
    (128,  *gpt_specs["1.3B"], *dummy_arguments, 8,   *fixed_params),
    (128,  *gpt_specs["1.3B"], *dummy_arguments, 16,  *fixed_params),
    (256,  *gpt_specs["1.3B"], *dummy_arguments, 16,  *fixed_params),
    (256,  *gpt_specs["1.3B"], *dummy_arguments, 32,  *fixed_params),
    (512,  *gpt_specs["1.3B"], *dummy_arguments, 32,  *fixed_params),
    (512,  *gpt_specs["1.3B"], *dummy_arguments, 64,  *fixed_params),
    (1024, *gpt_specs["1.3B"], *dummy_arguments, 32,  *fixed_params),
    (1024, *gpt_specs["1.3B"], *dummy_arguments, 64,  *fixed_params),
    (1024, *gpt_specs["1.3B"], *dummy_arguments, 128, *fixed_params),
],

8: [
    (16,   *gpt_specs["1.3B"], *dummy_arguments, 1,   *fixed_params),
    (32,   *gpt_specs["1.3B"], *dummy_arguments, 1,   *fixed_params),
    (32,   *gpt_specs["1.3B"], *dummy_arguments, 2,   *fixed_params),
    (64,   *gpt_specs["1.3B"], *dummy_arguments, 2,   *fixed_params),
    (64,   *gpt_specs["1.3B"], *dummy_arguments, 4,   *fixed_params),
    (128,  *gpt_specs["1.3B"], *dummy_arguments, 4,   *fixed_params),
    (128,  *gpt_specs["1.3B"], *dummy_arguments, 8,   *fixed_params),
    (256,  *gpt_specs["1.3B"], *dummy_arguments, 8,   *fixed_params),
    (256,  *gpt_specs["1.3B"], *dummy_arguments, 16,  *fixed_params),
    (512,  *gpt_specs["1.3B"], *dummy_arguments, 16,  *fixed_params),
    (512,  *gpt_specs["1.3B"], *dummy_arguments, 32,  *fixed_params),
    (1024, *gpt_specs["1.3B"], *dummy_arguments, 8,   *fixed_params),
    (1024, *gpt_specs["1.3B"], *dummy_arguments, 32,  *fixed_params),
    (1024, *gpt_specs["1.3B"], *dummy_arguments, 64,  *fixed_params),
    (1024, *gpt_specs["1.3B"], *dummy_arguments, 128, *fixed_params),

    (8,    *gpt_specs["2.7B"], *dummy_arguments, 1, *fixed_params),
    (16,   *gpt_specs["2.7B"], *dummy_arguments, 1, *fixed_params),
    (16,   *gpt_specs["2.7B"], *dummy_arguments, 2, *fixed_params),
    (32,   *gpt_specs["2.7B"], *dummy_arguments, 2, *fixed_params),
    (32,   *gpt_specs["2.7B"], *dummy_arguments, 4, *fixed_params),
    (64,   *gpt_specs["2.7B"], *dummy_arguments, 4, *fixed_params),
    (64,   *gpt_specs["2.7B"], *dummy_arguments, 8, *fixed_params),
    (128,  *gpt_specs["2.7B"], *dummy_arguments, 8, *fixed_params),
    (128,  *gpt_specs["2.7B"], *dummy_arguments, 16, *fixed_params),
    (256,  *gpt_specs["2.7B"], *dummy_arguments, 16, *fixed_params),
    (256,  *gpt_specs["2.7B"], *dummy_arguments, 32, *fixed_params),
    (512,  *gpt_specs["2.7B"], *dummy_arguments, 32, *fixed_params),
    (512,  *gpt_specs["2.7B"], *dummy_arguments, 64, *fixed_params),
    (1024, *gpt_specs["2.7B"], *dummy_arguments, 32, *fixed_params),
    (1024, *gpt_specs["2.7B"], *dummy_arguments, 64, *fixed_params),
    (1024, *gpt_specs["2.7B"], *dummy_arguments, 128, *fixed_params),
],
}
