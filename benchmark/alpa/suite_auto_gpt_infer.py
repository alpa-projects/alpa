"""Benchmark suites for gpt with auto parallelization."""
from suite_manual_gpt import gpt_specs
import numpy as np
import pprint

auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "all",
    "auto_stage_imbalance_tolerance": 1.0,
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}

prefer_reduce_scatter = True
use_remat = False


def get_solution_case(model_name, num_micro_batches, max_global_batch_size, num_auto_layers,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts):
    return [(model_name, max_global_batch_size, *gpt_specs[model_name], num_micro_batches,
             "load_solution",
             (prefer_reduce_scatter, use_remat, num_auto_layers,
              (forward_stage_layer_ids, submesh_physical_shapes,
               submesh_logical_shapes, submesh_autosharding_option_dicts)))]

perf_test_suite = {}
model_sizes = list(gpt_specs.keys())
model_size = model_sizes[0] 
num_micro_batch_config = [1, 4, 16, 64, 256]
batch_size_config = [1, 16, 64, 256, 1024]

for num_hosts in [1, 2, 4]: 
    for num_devices_per_host in [1, 2, 4, 8]:
        num_gpus = num_hosts * num_devices_per_host
        device_config = (num_hosts, num_devices_per_host) 
        if perf_test_suite.get(device_config) is None:
            perf_test_suite[device_config] = []
        for stage_num in [2**i for i in range(int(np.log2(num_hosts*num_devices_per_host))+1)]:
            if num_hosts < stage_num:
                num_hosts_per_stage = 1 
                num_devices_per_host_per_stage = num_devices_per_host // (stage_num // num_hosts)
            else:
                num_hosts_per_stage = num_hosts // stage_num
                num_devices_per_host_per_stage = num_devices_per_host
            num_gpus_per_stage = num_gpus // stage_num
            for data_parallel in [2**i for i in range(int(np.log2(num_gpus_per_stage))+1)]:
                for batch_size in batch_size_config:
                    for num_micro_batch in num_micro_batch_config:
                        max_global_batch_size = batch_size * num_micro_batch 
                        perf_test_suite[device_config] += \
                            get_solution_case(model_size, num_micro_batch, max_global_batch_size, stage_num, [[i] for i in range(stage_num)], 
                                            [(num_hosts_per_stage, num_devices_per_host_per_stage)] * stage_num,
                                            [(data_parallel, num_gpus_per_stage // data_parallel)] * stage_num,
                                            [{'force_batch_dim_to_mesh_dim': 0}] * stage_num)

#pprint.pprint(perf_test_suite)

# Temporary debug suite
tmp_suite = {}

# Performance test with search solutions found for p3.16xlarge
# perf_test_suite = {
#     (1, 1): get_solution_case("350M", 512,
#                          1, [[0]],
#                          [(1, 1)], [(1, 1)],
#                          [{}]),
    # 2: get_solution_case("760M", 128,
    #                      2, [[0], [1]],
    #                      [(1, 1)] * 2, [(1, 1)] * 2,
    #                      [{'force_batch_dim_to_mesh_dim': 0}] * 2),
    # 8:
    #     get_solution_case("1.3B", 64, 4, [[0], [1], [2], [3]], [(1, 2)] * 4,
    #                       [(1, 2)] * 4, [{
    #                           'force_batch_dim_to_mesh_dim': 0
    #                       }] * 4),

    # 8: get_solution_case("1.3B", 64,
    #                      8, [[0], [1], [2], [3], [4], [5], [6], [7], [8]],
    #                      [(1, 2)] * 4, [(2, 1)] * 4,
    #                      [{'force_batch_dim_to_mesh_dim': 0}] * 4),
    # 8: get_solution_case("2.6B", 128,
    #                      8, [[0, 1], [2, 3], [4, 5, 6, 7]],
    #                      [(1, 2), (1, 2), (1, 4)], [(2, 1), (2, 1), (4, 1)],
    #                      [{'force_batch_dim_to_mesh_dim': 0}, {}, {}]),
    # 16: get_solution_case("6.7B", 64,
    #                       8, [[0, 1, 2, 3], [4, 5, 6, 7]],
    #                       [(1, 8)] * 2, [(2, 4)] * 2,
    #                       [{'force_batch_dim_to_mesh_dim': 0}] * 2),
    # 32: get_solution_case("15B", 128,
    #                       16, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    #                       [(1, 8)] * 4, [(2, 4)] * 4,
    #                       [{'force_batch_dim_to_mesh_dim': 0}] * 4),
    # 64: get_solution_case("39B", 1024,
    #                       16, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
    #                       [(1, 4)] * 16, [(1, 4)] * 16,
    #                       [{'force_batch_dim_to_mesh_dim': 0}] * 16),
#}
