from collections import namedtuple
import time

import jax
import ray

from parax import DeviceCluster, global_config
from benchmark_gpt_bert import benchmark_gpt_bert_internal
from parax.util import write_tsv, to_str_round


gpt_auto_sharding = [
    # B,  S,     H,    L,  #head,     V,     D0, D1, NB, FD,    RS,   CK
    (8,   1024,  2048, 10, 2048//128, 25600, 1,  1,  1,  False, True, False),
    (8,   1024,  3072, 10, 3072//128, 25600, 1,  2,  1,  False, True, False),
    (8,   1024,  4096, 10, 4096//128, 25600, 1,  4,  1,  False, True, False),
    (8,   1024,  5760, 10, 5760//128, 25600, 1,  8,  1,  False, True, False),
]

gpt_data_parallel = [
    # B,  S,     H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    (8,   1024,  2048, 10, 2048//128, 25600, 1,  1,  1,  True,  False, False),
    (8,   1024,  3072, 10, 3072//128, 25600, 1,  2,  1,  True,  False, False),
    (8,   1024,  4096, 10, 4096//128, 25600, 1,  4,  1,  True,  False, False),
    (8,   1024,  5760, 10, 5760//128, 25600, 1,  8,  1,  True,  False, False),
]

gpt_zero_2 = [
    # B,  S,     H,    L,  #head,     V,     D0, D1, NB, FD,    RS,   CK
    (8,   1024,  2048, 10, 2048//128, 25600, 1,  1,  1,  True,  True, False),
    (8,   1024,  3072, 10, 3072//128, 25600, 1,  2,  1,  True,  True, False),
    (8,   1024,  4096, 10, 4096//128, 25600, 1,  4,  1,  True,  True, False),
    (8,   1024,  5760, 10, 5760//128, 25600, 1,  8,  1,  True,  True, False),
]


Case = namedtuple("Case", ["exp_name", "instance", "num_nodes", "num_gpus_per_node",
                           "model_name", "method", "func", "args"])


def benchmark_gpt_one_case(case):
    device_cluster = DeviceCluster()
    physical_mesh = device_cluster.get_physical_mesh(
        list(range(case.num_nodes)), case.num_gpus_per_node)

    result = benchmark_gpt_bert_internal(physical_mesh, "gpt", case.args, 5)
    latencies, alloc_mem, tflops, param_count, ilp_objective = result
    value_dict = {
        "latencies": latencies,
        "tflops": tflops,
        "alloc_mem": alloc_mem / (1024 ** 3),
        "param_count": param_count / 1e9,
    }

    # Log results
    heads = ["Exp", "Instance", "num_nodes", "num_gpus_per_node", "model_name", 
             "method", "value", "tstamp"]
    values = [case.exp_name, case.instance, case.num_nodes, case.num_gpus_per_node,
              case.model_name, case.method, to_str_round(value_dict, 4),
              int(time.time())]
    write_tsv(heads, values, f"result.tsv")

    physical_mesh.shutdown()


def build_cases():
    instance = "p3.24xlarge"
    exp_name = "weak_scaling_model"
    num_gpus_list = [1, 2, 4, 8]

    suites = [
        #("GPT", "parax.auto_sharding", gpt_auto_sharding, benchmark_gpt_one_case),
        ("GPT", "parax.data_parallel", gpt_data_parallel, benchmark_gpt_one_case),
        #("GPT", "parax.zero_2", gpt_zero_2, benchmark_gpt_one_case),
    ]

    cases = []
    for suite in suites:
        model_name, method, args_list, benchmark_func = suite
        for i, args in enumerate(args_list):
            num_gpus = num_gpus_list[i]
            num_nodes = ((num_gpus + 7) // 8)
            num_gpus_per_node = min(num_gpus, 8)
            cases.append(Case(exp_name, instance, num_nodes, num_gpus_per_node,
                              model_name, method, benchmark_func, args))

    return cases


if __name__ == "__main__":
    ray.init(address="auto")
    jax.config.update('jax_platform_name', 'cpu')
    global_config.use_dummy_value_for_benchmarking = True

    cases = build_cases()

    for case in cases:
        case.func(case)
