"""Suites for wresnet benchmarking."""
_ = None

# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor,
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping,
# RS = prefer_reduce_scatter, Remat = use_rematerialization
# LS = logical_mesh_search_space

wresnet_specs = {
    #    I,   L,  C,   W,  dtype,  
"250M": (224, 50, 160, 2,  "fp32"), 
"500M": (224, 50, 224, 2,  "fp32"), 
"1B":   (224, 50, 320, 2,  "fp32"), 
"2B":   (224, 50, 448, 2,  "fp32"), 
"4B":   (224, 50, 640, 2,  "fp32"), 
"6.8B": (224, 50, 320, 16, "fp32"), 
"13B":  (224, 50, 320, 32, "fp32"),
}

_ = None

fast_perf_test_wresnet_suite = { # key = the number of gpus, value = a list of cases
    1: [
        #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FM,    RS,    Remat, other
        (16,   224, 50,  192, 2, "fp32", 1,  1,  1,  False, True,  False, _),
    ],

    4 : [
        #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FM,    RS,    Remat, other
        (32,   224, 50,  320, 2, "fp32", 1,  4,  1,  False, False, False, _),
    ],

    8: [
        #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FM,    RS,    Remat, other
        (64,   224, 50,  320, 2, "fp32", 1,  8,  1,  False, False, False, _),
    ],
}


paper_auto_wresnet_suite = {  # key = the number of gpus, value = a list of cases
    1: [
        # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
        (1536, 224, 50,  160,  2,  "fp32", 24, False, False, True,  "single_node_model_parallel"),
    ],

    2: [
        # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
        (1536, 224, 50,  224,  2, "fp32",  24, False, True,  True,  "single_node_model_parallel"),
    ],

    4 : [
        # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
        (1536,  224, 50,  320,  2, "fp32",  24, False, True,  True,  "single_node_model_parallel"),
    ],

    8: [
        # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
        (1536, 224, 50,  448,  2, "fp32",  24, False, True,  True,  "single_node_model_parallel"),
    ],

    16: [
        # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
        (1536, 224, 50,  640,  2,  "fp32", 32, False, True,  True,  "single_node_model_parallel"),
    ],

    32: [
        # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
        #(1536, 224, 50,  640,  2,  "fp32", 32, False, True,  True, "single_node_model_parallel"),
        (1520, 224, 50,  320,  16, "fp32", 38, False, False, True,  "single_node_model_parallel"),
    ],

    64: [
        # B,   I,   L,   C,    W,  dtype,  NB, FD,    RS,    Remat, LS
        # (1536, 224, 50,  320,  32, "fp32", 32, False, False, True,  "single_node_model_parallel"),
        # (1520, 224, 50,  320,  32, "fp32", 38, False, False, True,  "single_node_model_parallel"),
        # (1536, 224, 101,  320,  16, "fp32", 48, False, False, True,  "single_node_model_parallel"),
        (1520, 224, 101,  320,  16, "fp32", 38, False, False, True,  "single_node_model_parallel"),
        # (1536, 224, 50,  320,  32, "fp32", 48, False, False, True,  "single_node_model_parallel"),
        # (1536, 224, 50,  320,  32, "fp32", 48, False, False, True,  "single_node_model_parallel"),
    ],
}