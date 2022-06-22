"""Benchmark suites for gpt with auto parallelization."""
from suite_manual_gpt import gpt_specs

auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "all",
    "auto_stage_imbalance_tolerance": 1.0,
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}

prefer_reduce_scatter = True
use_remat = False


def get_solution_case(model_name, num_micro_batches, max_global_batch_size,
                      num_auto_layers, forward_stage_layer_ids,
                      submesh_physical_shapes, submesh_logical_shapes,
                      submesh_autosharding_option_dicts):
    return [(model_name, max_global_batch_size, *gpt_specs[model_name],
             num_micro_batches, "load_solution",
             (prefer_reduce_scatter, use_remat, num_auto_layers,
              (forward_stage_layer_ids, submesh_physical_shapes,
               submesh_logical_shapes, submesh_autosharding_option_dicts)))]


perf_test_suite = {}
model_sizes = [
    "125M", "350M", "760M", "1.3B", "2.6B", "6.7B", "15B", "39B", "76B"
]
model_size = model_sizes[4]
#num_micro_batch_config = [1, 4, 16, 64, 256]
#batch_size_config = [1, 4, 16, 64]
num_micro_batch_config = [1]
batch_size_config = [1]


def get_config(pp_list,
               dp_list,
               op_list,
               ignore_one_device_case=False,
               debug=False):
    for num_hosts in [1]:
        for num_devices_per_host in [1, 2, 4, 8]:
            if ignore_one_device_case and num_hosts == 1 and num_devices_per_host == 1:
                continue
            num_gpus = num_hosts * num_devices_per_host
            device_config = (num_hosts, num_devices_per_host)
            if perf_test_suite.get(device_config) is None:
                perf_test_suite[device_config] = []
            for stage_num in pp_list:
                if stage_num > num_gpus:
                    continue
                if num_hosts < stage_num:
                    num_hosts_per_stage = 1
                    num_devices_per_host_per_stage = num_devices_per_host // (
                        stage_num // num_hosts)
                else:
                    num_hosts_per_stage = num_hosts // stage_num
                    num_devices_per_host_per_stage = num_devices_per_host
                num_gpus_per_stage = num_gpus // stage_num
                for data_parallel in dp_list:
                    for operator_parallel in op_list:
                        if data_parallel * operator_parallel != num_gpus_per_stage:
                            continue
                        for batch_size in batch_size_config:
                            for num_micro_batch in num_micro_batch_config:
                                if debug:
                                    print(
                                        f"({batch_size}, {num_micro_batch}) ({stage_num}, {data_parallel}, {operator_parallel})"
                                    )
                                max_global_batch_size = batch_size * num_micro_batch
                                perf_test_suite[device_config] += \
                                    get_solution_case(model_size, num_micro_batch, max_global_batch_size, stage_num, [[i] for i in range(stage_num)],
                                                    [(num_hosts_per_stage, num_devices_per_host_per_stage)] * stage_num,
                                                    [(data_parallel, operator_parallel)] * stage_num,
                                                    [{'force_batch_dim_to_mesh_dim': 0}] * stage_num)


if __name__ == "__main__":
    #get_config([1, 2, 4, 8], [1], [1], False, True)
    #get_config([1], [1, 2, 4, 8], [1], True, True)
    #get_config([1], [1], [1, 2, 4, 8], True, True)
    get_config([1], [1], [4], True, True)
else:
    #get_config([1, 2, 4, 8], [1], [1], False)
    #get_config([1], [1, 2, 4, 8], [1], True)
    #get_config([1], [1], [1, 2, 4, 8], True)
    get_config([1], [1], [4], True, False)
