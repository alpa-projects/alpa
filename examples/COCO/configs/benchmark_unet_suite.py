"""Benchmark suite for unet on coco dataset."""

# Meanings for each column from left to right.
# number of gpus : (batch_size, image_size, num_micro_batches, channel_size_choices, block_cnt_choices, force_batch_dim_mapping, prefer_reduce_scatter, use_remat, logical_mesh_search_space)
unet_suite = {
    1: [
        (256, (384, 384), 2, (16, 32, 64, 128), (4, 4, 4, 4, 4), False, False, True,  "single_node_model_parallel"),
    ],
    2: [
        (256, (384, 384), 2, (48, 64, 96, 128), (6, 6, 6, 6, 6), False, False, True, 'single_node_model_parallel'),
    ],
    4: [
        (256, (384, 384), 4, (64, 128, 256, 256), (10, 10, 10, 10, 10), False, False, True,  "single_node_model_parallel" ),
    ], 
    8: [
        (64, (384, 384), 4, (32, 64, 128, 256), (8, 8, 8, 8, 8), False, False, True,  "all" ),
    ], 
    16: [
        (128, (384, 384), 4, (64, 64, 128, 128), (8, 8, 8, 8, 8), False, False, True,  "all" ),
    ],
    32: [
        (256, (384, 384), 4, (64, 64, 128, 128), (10, 10, 10, 10, 10), False, False, True,  "all" ),
    ]
}
