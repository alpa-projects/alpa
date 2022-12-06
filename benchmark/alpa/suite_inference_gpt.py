"""Benchmark suites for gpt with auto parallelization."""
from alpa import AutoStageOption
from suite_manual_gpt import gpt_specs
from benchmark_parallel_utils import (BenchmarkCase, UniformParallelArgs,
                                      LoadSolutionParallelArgs,
                                      SearchParallelArgs)

prefer_reduce_scatter = True
force_batch_dim_mapping = True
use_remat = False

profile_suite = {}
force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}


def get_config(model_config,
               pp_list,
               dp_list,
               op_list,
               num_micro_batch_config,
               batch_size_config,
               ignore_one_device_case=False):
    for pp in pp_list:
        for dp in dp_list:
            for op in op_list:
                num_gpus = pp * dp * op
                if ignore_one_device_case and num_gpus == 1:
                    continue
                for bs in batch_size_config:
                    for nb in num_micro_batch_config:
                        total_bs = bs * nb
                        if num_gpus not in profile_suite:
                            profile_suite[num_gpus] = []
                        parallel_args = UniformParallelArgs(
                            prefer_reduce_scatter, use_remat, dp, op, pp,
                            force_batch_dim_mapping)
                        case = BenchmarkCase(total_bs, model_config, nb,
                                             "uniform", parallel_args)
                        profile_suite[num_gpus].append(case)


## general examples:
#get_config(gpt_specs["350M"], [1, 2, 4, 8], [1], [1], [1], [1, 4, 16])
#get_config(gpt_specs["760M"], [1, 2, 4, 8], [1], [1], [1], [1, 4, 16])
#get_config(gpt_specs["1.3B"], [1, 2, 4, 8], [1], [1], [1], [1, 4, 16])
#get_config(gpt_specs["2.6B"], [1, 2, 4, 8], [1], [1], [1], [1, 4, 16])
#get_config(gpt_specs["6.7B"], [1, 2, 4, 8], [1], [1], [1], [1, 4, 16])
#get_config(gpt_specs["15B"],  [1, 2, 4, 8], [1], [1], [1], [1, 4, 16])

## benchmark specific parallel method:
#get_config(gpt_specs["6.7B"], [1], [1], [1, 2, 4, 8], [1, 256], [1, 4, 16, 64])
#get_config(gpt_specs["6.7B"], [1], [1, 2, 4, 8], [1], [1, 256], [1, 4, 16, 64],
#           ignore_one_device_case=True)
#get_config(gpt_specs["6.7B"], [1, 2, 4, 8], [1], [1], [1, 256], [1, 4, 16, 64],
#           ignore_one_device_case=True)

## generate inference profiling results
get_config(gpt_specs["1.3B"], [1, 2, 4, 8], [1], [1, 2, 4, 8], [1],
           [1, 2, 4, 8, 16])
get_config(gpt_specs["2.6B"], [1, 2, 4, 8, 16, 32], [1], [1, 2, 4, 8], [1],
           [1, 2, 4, 8, 16])
get_config(gpt_specs["6.7B"], [1, 2, 4, 8, 16, 32], [1], [1, 2, 4, 8], [1],
           [1, 2, 4, 8, 16])
get_config(gpt_specs["15B"], [1, 2, 4, 8, 16], [1], [1, 2, 4, 8], [1],
           [1, 2, 4, 8, 16])

test_suite = {
    8: [
        BenchmarkCase(
            1, gpt_specs["1.3B"], 1, "uniform",
            UniformParallelArgs(
                prefer_reduce_scatter,
                use_remat,
                dp=1,
                op=1,
                pp=8,
                force_batch_dim_mapping=force_batch_dim_mapping)),
        BenchmarkCase(
            1, gpt_specs["1.3B"], 1, "load_solution",
            LoadSolutionParallelArgs(
                prefer_reduce_scatter,
                use_remat,
                num_auto_layers=8,
                forward_stage_layer_ids=[[0], [1], [2], [3], [4], [5], [6],
                                         [7]],
                submesh_physical_shapes=[(1, 1)] * 8,
                submesh_logical_shapes=[(1, 1)] * 8,
                submesh_autosharding_option_dicts=[force_dp_dict] * 8)),
        # 2D + Profile
        BenchmarkCase(
            1, gpt_specs["1.3B"], 1, "load_solution",
            LoadSolutionParallelArgs(
                prefer_reduce_scatter,
                use_remat,
                num_auto_layers=50,
                forward_stage_layer_ids=[[0, 1, 2, 3, 4, 5],
                                         [6, 7, 8, 9, 10, 11],
                                         [12, 13, 14, 15, 16, 17, 18],
                                         [19, 20, 21, 22, 23, 24, 25],
                                         [26, 27, 28, 29, 30, 31],
                                         [32, 33, 34, 35, 36, 37],
                                         [38, 39, 40, 41, 42, 43, 44],
                                         [45, 46, 47, 48, 49]],
                submesh_physical_shapes=[(1, 1)] * 8,
                submesh_logical_shapes=[(1, 1)] * 8,
                submesh_autosharding_option_dicts=[force_dp_dict] * 8)),
        # 1D + Profile
        BenchmarkCase(
            1, gpt_specs["1.3B"], 1, "load_solution",
            LoadSolutionParallelArgs(
                prefer_reduce_scatter,
                use_remat,
                num_auto_layers=50,
                forward_stage_layer_ids=[[0, 1, 2, 3, 4, 5],
                                         [6, 7, 8, 9, 10, 11, 12],
                                         [13, 14, 15, 16, 17, 18],
                                         [19, 20, 21, 22, 23, 24],
                                         [25, 26, 27, 28, 29, 30, 31],
                                         [32, 33, 34, 35, 36, 37],
                                         [38, 39, 40, 41, 42, 43, 44],
                                         [45, 46, 47, 48, 49]],
                submesh_physical_shapes=[(1, 1)] * 8,
                submesh_logical_shapes=[(1, 1)] * 8,
                submesh_autosharding_option_dicts=[force_dp_dict] * 8)),
        # 1D + Cost model
        BenchmarkCase(
            1, gpt_specs["1.3B"], 1, "load_solution",
            LoadSolutionParallelArgs(
                prefer_reduce_scatter,
                use_remat,
                num_auto_layers=50,
                forward_stage_layer_ids=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10],
                                         [11, 12, 13, 14, 15, 16],
                                         [17, 18, 19, 20, 21, 22, 23],
                                         [24, 25, 26, 27, 28, 29, 30],
                                         [31, 32, 33, 34, 35, 36, 37],
                                         [38, 39, 40, 41, 42, 43, 44],
                                         [45, 46, 47, 48, 49]],
                submesh_physical_shapes=[(1, 1)] * 8,
                submesh_logical_shapes=[(1, 1)] * 8,
                submesh_autosharding_option_dicts=[force_dp_dict] * 8)),
    ]
}

search_suite = {}


def generate_search_configs(model_config, num_auto_layers, pp_list, op_list):
    """Generate search configs."""
    for pp in pp_list:
        for op in op_list:
            num_gpus = pp * op
            if num_gpus not in search_suite:
                search_suite[num_gpus] = []
            search_suite[num_gpus].append(
                BenchmarkCase(
                    1,
                    model_config,
                    1,
                    "search",
                    SearchParallelArgs(
                        prefer_reduce_scatter,
                        use_remat,
                        num_auto_layers=num_auto_layers,
                        auto_stage_option={
                            "submesh_physical_shape_space":
                                "manual",
                            "manually_specified_submeshes": ((1, op),),
                            "submesh_logical_shape_space":
                                "model_parallel_only",
                            "layer_profile_mode":
                                "individual",
                            # "use_hlo_cost_model": True,
                            # "profiling_database_filename":
                            #   "prof_database.pkl",
                        })))


generate_search_configs(gpt_specs["1.3B"], 50, [8], [1])
# generate_search_configs(gpt_specs["1.3B"], 50, [1, 2, 4, 8, 16, 32],
#                         [1, 2, 4, 8])
# generate_search_configs(gpt_specs["2.6B"], 66, [1, 2, 4, 8, 16, 32],
#                         [1, 2, 4, 8])
# generate_search_configs(gpt_specs["6.7B"], 66, [1, 2, 4, 8, 16, 32],
#                         [1, 2, 4, 8])
