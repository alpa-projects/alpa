"""Benchmark suite for unet on coco dataset."""

# Meanings for each column from left to right.
# number of gpus : (batch_size, image_size, num_micro_batches, channel_size_choices, block_cnt_choices, force_batch_dim_mapping, prefer_reduce_scatter, use_remat, logical_mesh_search_space)
unet_suite = {
    1: [
        (256, (384, 384), 2, (16, 16, 16, 16), (4, 4, 4, 4, 4), False, False, True,  "single_node_model_parallel"),
        ],
    2: [ 
        (512, (384, 384), 8, (16, 32, 32, 32), (4, 4, 4, 4, 4), False, False, True, 'single_node_model_parallel'),
    ],
    4: [
        (512, (384, 384), 4, (32, 32, 48, 48), (4, 4, 4, 4, 4), False, False, True,  "single_node_model_parallel"),
    ], 
    8: [
        (512, (384, 384), 4, (48, 48, 64, 64), (4, 4, 4, 4, 4), False, False, True,  "all" ),
    ], 
    16: [
        (512, (384, 384), 8, (64, 64, 64, 64), (4, 6, 6, 6, 10), False, False, True,  "all" ),
    ],
    32: [
        (512, (384, 384), 8, (64, 64, 96, 96), (6, 8, 8, 8, 8), False, False, True,  "all" ),
    ]
}
