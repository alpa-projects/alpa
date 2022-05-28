"""Benchmark suite for auto wide-ResNet"""
from benchmark.alpa.suite_paper_wresnet import get_auto_test_case, wresnet_specs

_ = None

def copy_cases_with_search_space(cases, search_space, num_gpu):
    new_cases = []
    for case in cases:
        case = list(case)
        overwrite_global_config = case[-1] = dict(case[-1])
        if search_space == "ppdp":
            overwrite_global_config["logical_mesh_search_space"] = "dp_only"
        elif search_space == "inter-only":
            overwrite_global_config["fix_physical_mesh_shape"] = (1, 1)
        else:
            raise RuntimeError("Unsupported search space: " + search_space)
        new_cases.append(case)
    return new_cases

def _num_gpu_to_mesh_shape(num_gpu):
    if num_gpu <= 8:
        return (1, num_gpu)
    assert num_gpu % 8 == 0
    return (num_gpu // 8, 8)


artifact_search_e2e_wresnet_suite = {  # key = the number of gpus, value = a list of cases
1: get_auto_test_case("250M", [24], 1536),
4: get_auto_test_case("1B", [24], 1536),
8: get_auto_test_case("2B", [24], 1536),
16: get_auto_test_case("4B", [32], 1536),
32: get_auto_test_case("6.8B", [38], 1520),
}


artifact_search_e2e_wresnet_ppdp_suite = {
num_gpus: copy_cases_with_search_space(artifact_search_e2e_wresnet_suite[num_gpus], "ppdp", num_gpus)
for num_gpus in artifact_search_e2e_wresnet_suite
}
# Manually reduce the microbatch size to make it solvable.
artifact_search_e2e_wresnet_ppdp_suite[4][0][6] = 32
artifact_search_e2e_wresnet_ppdp_suite[8][0][6] = 32
artifact_search_e2e_wresnet_ppdp_suite[16][0][6] = 384


artifact_search_e2e_wresnet_inter_only_suite = {
num_gpus: copy_cases_with_search_space(artifact_search_e2e_wresnet_suite[num_gpus], "inter-only", num_gpus)
for num_gpus in artifact_search_e2e_wresnet_suite
}
artifact_search_e2e_wresnet_inter_only_suite[4][0][6] = 64
artifact_search_e2e_wresnet_inter_only_suite[8][0][6] = 96
artifact_search_e2e_wresnet_inter_only_suite[16][0][6] = 384

artifact_result_e2e_wresnet_suite = {
1: get_auto_test_case("250M", [24], 1536, overwrite_global_config_dict={
    "strategy": "shard_parallel"
}),
2: get_auto_test_case("500M", [24], 1536, overwrite_global_config_dict={
    "strategy": "shard_parallel"
}),
4: get_auto_test_case("1B", [24], 1536, overwrite_global_config_dict={
    "strategy": "shard_parallel"
}),
8: get_auto_test_case("2B", [24], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 4)],
    "sub_logical_mesh_shapes": [(4, 1), (1, 4)],
    "submesh_autosharding_option_dicts": [{}, {'force_batch_dim_to_mesh_dim': 0}]
}),
16: get_auto_test_case("4B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 4), (1, 8)],
    "sub_logical_mesh_shapes": [(4, 1), (4, 1), (8, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}, {}]
}),
32: get_auto_test_case("6.8B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 8), (1, 8), (1, 8), (1, 8)],
    "sub_logical_mesh_shapes": [(8, 1), (8, 1), (8, 1), (8, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {}, {}, {}],
}),
}


artifact_e2e_wresnet_intra_only_suite = {
1:  [(1536, *wresnet_specs["250M"], 1,  1, 48, False, True, _, _)],
2:  [(1536, *wresnet_specs["500M"], 2,  1, 32, False, True, _, _)],
4:  [(1536, *wresnet_specs["1B"],   4,  1, 32, False, True, _, _)],
8:  [(1536, *wresnet_specs["2B"],   8,  1, 48, False, True, _, _)],
16: [(1536, *wresnet_specs["4B"],   2,  8, 64, False, True, _, _)],
32: [(1536, *wresnet_specs["6.8B"], 4,  8, 48, False, True, _, _)],
}


artifact_result_e2e_wresnet_inter_only_suite = {
1: get_auto_test_case("250M", [24], 1536, overwrite_global_config_dict={
    "strategy": "shard_parallel"
}),
4: get_auto_test_case("1B", [64], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13], [14, 15]],
    "sub_physical_mesh_shapes": [(1, 1)] * 4,
    "sub_logical_mesh_shapes": [(1, 1)] * 4,
    "submesh_autosharding_option_dicts": [{}] * 4
}),
8: get_auto_test_case("2B", [96], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9, 10], [11, 12], [13], [14], [15]],
    "sub_physical_mesh_shapes": [(1, 1)] * 8,
    "sub_logical_mesh_shapes": [(1, 1)] * 8,
    "submesh_autosharding_option_dicts": [{}] * 8
}),
16: get_auto_test_case("4B", [384], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[i] for i in range(16)],
    "sub_physical_mesh_shapes": [(1, 1)] * 16,
    "sub_logical_mesh_shapes": [(1, 1)] * 16,
    "submesh_autosharding_option_dicts": [{}] * 16
}),
# OOM even manually design a microbatch num
32: get_auto_test_case("6.8B", [2], 2, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[i] for i in range(32)],
    "sub_physical_mesh_shapes": [(1, 1)] * 32,
    "sub_logical_mesh_shapes": [(1, 1)] * 32,
    "submesh_autosharding_option_dicts": [{}] * 32
}),
}


artifact_result_e2e_wresnet_ppdp_suite = {
1: get_auto_test_case("250M", [24], 1536, overwrite_global_config_dict={
    "strategy": "shard_parallel"
}),
4: get_auto_test_case("1B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12], [13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 2), (1, 1), (1, 1)],
    "sub_logical_mesh_shapes": [(2, 1), (1, 1), (1, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 3
}),
8: get_auto_test_case("2B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10], [11, 12, 13], [14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 1), (1, 2), (1, 1)],
    "sub_logical_mesh_shapes": [(4, 1), (1, 1), (2, 1), (1, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 4
}),
16: get_auto_test_case("4B", [384], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[i] for i in range(16)],
    "sub_physical_mesh_shapes": [(1, 1)] * 16,
    "sub_logical_mesh_shapes": [(1, 1)] * 16,
    "submesh_autosharding_option_dicts": [{}] * 16
}),
# OOM, so manually design a microbatch num
32: get_auto_test_case("6.8B", [2], 2, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[i] for i in range(16)],
    "sub_physical_mesh_shapes": [(1, 2)] * 16,
    "sub_logical_mesh_shapes": [(2, 1)] * 16,
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 16,
}),
}


paper_inter_op_ablation_wresnet_search_suite = {
    8:  get_auto_test_case("2B", [32], 1536),

    16: get_auto_test_case("4B", [32], 1536),

    32: get_auto_test_case("6.8B", [32], 1536)
}


paper_inter_op_ablation_wresnet_result_suite = {
8: # Ours
get_auto_test_case("2B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 4)],
    "sub_logical_mesh_shapes": [(4, 1), (1, 4)],
    "submesh_autosharding_option_dicts": [{}, {'force_batch_dim_to_mesh_dim': 0}]
}) + # Equal operator
get_auto_test_case("2B", [32], 1536, overwrite_global_config_dict={
    "use_equal_eqn": True,
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4)] * 2,
    "sub_logical_mesh_shapes": [(2, 2), (1, 4)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2
}) + # Equal Layer
get_auto_test_case("2B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 4)],
    "sub_logical_mesh_shapes": [(4, 1), (1, 4)],
    "submesh_autosharding_option_dicts": [{}, {'force_batch_dim_to_mesh_dim': 0}]
}),
16: # Ours
get_auto_test_case("4B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 4), (1, 8)],
    "sub_logical_mesh_shapes": [(4, 1), (4, 1), (8, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}, {}]
}) + # Equal operator
get_auto_test_case("4B", [35], 1540, overwrite_global_config_dict={
    "use_equal_eqn": True,
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 4), (1, 4), (1, 8)],
    "sub_logical_mesh_shapes": [(4, 1), (1, 4), (1, 8)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 3
}) + # Equal layer
get_auto_test_case("4B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 8)] * 2,
    "sub_logical_mesh_shapes": [(2, 4), (1, 8)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 2
}),
32: # Ours
get_auto_test_case("6.8B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 8), (1, 8), (1, 8), (1, 8)],
    "sub_logical_mesh_shapes": [(8, 1), (8, 1), (8, 1), (8, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {}, {}, {}],
}) + # Equal operator
get_auto_test_case("6.8B", [35], 1540, overwrite_global_config_dict={
    "use_equal_eqn": True,
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 8), (1, 8), (1, 8), (1, 8)],
    "sub_logical_mesh_shapes": [(4, 2), (1, 8), (8, 1), (8, 1)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}, {}, {}],
}) + # Equal layer
get_auto_test_case("6.8B", [32], 1536, overwrite_global_config_dict={
    "pipeline_stage_mode": "manual_gpipe",
    "forward_stage_layer_ids": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    "sub_physical_mesh_shapes": [(1, 8), (1, 8), (1, 8), (1, 8)],
    "sub_logical_mesh_shapes": [(8, 1), (1, 8), (1, 8), (1, 8)],
    "submesh_autosharding_option_dicts": [{'force_batch_dim_to_mesh_dim': 0}] * 4,
}),
}