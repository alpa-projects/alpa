from collections import namedtuple
import time

import jax
import ray

from parax import DeviceCluster, global_config
from parax.util import write_tsv, to_str_round
from benchmark_gpt_bert import benchmark_gpt_bert_internal
from benchmark_wide_resnet import benchmark_wide_resnet_internal
from benchmark_moe import benchmark_moe_internal

benchmark_gpt_internal = lambda physical_mesh, args, niter : \
    benchmark_gpt_bert_internal(physical_mesh, "gpt", args, niter)

Case = namedtuple("Case", ["exp_name", "instance", "num_nodes", "num_gpus_per_node",
                           "model_name", "method", "func", "args"])


def benchmark_one_case(case):
    device_cluster = DeviceCluster()
    physical_mesh = device_cluster.get_physical_mesh(
        list(range(case.num_nodes)), case.num_gpus_per_node)

    result = case.func(physical_mesh, case.args, 5)
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
    write_tsv(heads, values, f"result_weak_scaling.tsv")

    physical_mesh.shutdown()


def build_cases():
    instance = "p3.24xlarge"
    exp_name = "weak_scaling_model"
    num_gpus_list = [1, 2, 4, 8]

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


gpt_auto_sharding = [
    # B,  S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    (8,   1024, 2048, 10, 2048//128, 25600, 1,  1,  1,  False, True,  False),
    (8,   1024, 3072, 10, 3072//128, 25600, 1,  2,  1,  False, True,  False),
    (8,   1024, 4096, 10, 4096//128, 25600, 1,  4,  1,  False, True,  False),
    (8,   1024, 6144, 10, 6144//128, 25600, 1,  8,  1,  False, True,  False),
]

gpt_data_parallel = [
    # B,  S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    (8,   1024, 2048, 10, 2048//128, 25600, 1,  1,  1,  True,  False, False),
    (8,   1024, 3072, 10, 3072//128, 25600, 1,  2,  1,  True,  False, False),
    (8,   1024, 4096, 10, 4096//128, 25600, 1,  4,  1,  True,  False, False),
    (8,   1024, 6144, 10, 6144//128, 25600, 1,  8,  1,  True,  False, False),
]

gpt_zero_2 = [
    # B,  S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    (8,   1024, 2048, 10, 2048//128, 25600, 1,  1,  1,  True,  True,  False),
    (8,   1024, 3072, 10, 3072//128, 25600, 1,  2,  1,  True,  True,  False),
    (8,   1024, 4096, 10, 4096//128, 25600, 1,  4,  1,  True,  True,  False),
    (8,   1024, 6144, 10, 6144//128, 25600, 1,  8,  1,  True,  True,  False),
]

w_resnet_auto_sharding = [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    CK,
    (32,   224, 50,  224, 4, "fp32", 1,  1,  1,  False, True,  False),
    (32,   224, 50,  320, 4, "fp32", 1,  2,  1,  False, True,  False),
    (32,   224, 50,  448, 4, "fp32", 1,  4,  1,  False, True,  False),
    (32,   224, 50,  640, 4, "fp32", 1,  8,  1,  False, True,  False),
]

w_resnet_data_parallel = [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    CK,
    (32,   224, 50,  224, 4, "fp32", 1,  1,  1,  True,  False, False),
    (32,   224, 50,  320, 4, "fp32", 1,  2,  1,  True,  False, False),
    (32,   224, 50,  448, 4, "fp32", 1,  4,  1,  True,  False, False),
    (32,   224, 50,  640, 4, "fp32", 1,  8,  1,  True,  False, False),
]

w_resnet_zero_2 = [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    CK,
    (32,   224, 50,  224, 4, "fp32", 1,  1,  1,  True,  True, False),
    (32,   224, 50,  320, 4, "fp32", 1,  2,  1,  True,  True, False),
    (32,   224, 50,  448, 4, "fp32", 1,  4,  1,  True,  True, False),
    (32,   224, 50,  640, 4, "fp32", 1,  8,  1,  True,  True, False),
]

moe_auto_sharding = [
    #B,   S,    H,    L,  #head,     V,     S_,   E,  D0, D1, NB, FD,    RS,    CK
    (16,  1024, 768,  12, 768//128,  25600, 1024, 16, 1,  1,  1,  False, True, False),
    (16,  1024, 1280, 12, 1280//128, 25600, 1024, 16, 1,  2,  1,  False, True, False),
    (16,  1024, 1920, 12, 1920//128, 25600, 1024, 16, 1,  4,  1,  False, True, False),
    (16,  1024, 2560, 12, 2560//128, 25600, 1024, 16, 1,  8,  1,  False, True, False),
]

moe_data_parallel = [
    #B,   S,    H,    L,  #head,     V,     S_,   E,  D0, D1, NB, FD,    RS,    CK
    (16,  1024, 768, 12,  768//128,  25600, 1024, 16, 1,  1,  1,  True,  False, False),
    (16,  1024, 1280, 12, 1280//128, 25600, 1024, 16, 1,  2,  1,  True,  False, False),
    (16,  1024, 1920, 12, 1920//128, 25600, 1024, 16, 1,  4,  1,  True,  False, False),
    (16,  1024, 2560, 12, 2560//128, 25600, 1024, 16, 1,  8,  1,  True,  False, False),
]

moe_zero_2 = [
    #B,   S,    H,    L,  #head,     V,     S_,   E,  D0, D1, NB, FD,    RS,    CK
    (16,  1024, 768,  12, 768//128,  25600, 1024, 16, 1,  1,  1,  True,  True,  False),
    (16,  1024, 1280, 12, 1280//128, 25600, 1024, 16, 1,  2,  1,  True,  True,  False),
    (16,  1024, 1920, 12, 1920//128, 25600, 1024, 16, 1,  4,  1,  True,  True,  False),
    (16,  1024, 2560, 12, 2560//128, 25600, 1024, 16, 1,  8,  1,  True,  True,  False),
]

suites = [
    ("GPT", "parax.auto_sharding", gpt_auto_sharding, benchmark_gpt_internal),
    ("GPT", "parax.data_parallel", gpt_data_parallel, benchmark_gpt_internal),
    ("GPT", "parax.zero_2", gpt_zero_2, benchmark_gpt_internal),
    ("W-ResNet", "parax.auto_sharding", w_resnet_auto_sharding, benchmark_wide_resnet_internal),
    ("W-ResNet", "parax.data_parallel", w_resnet_data_parallel, benchmark_wide_resnet_internal),
    ("W-ResNet", "parax.zero_2", w_resnet_zero_2, benchmark_wide_resnet_internal),
    ("MoE", "parax.auto_sharding", moe_auto_sharding, benchmark_moe_internal),
    ("MoE", "parax.data_parallel", moe_data_parallel, benchmark_moe_internal),
    ("MoE", "parax.zero_2", moe_zero_2, benchmark_moe_internal),
]


if __name__ == "__main__":
    ray.init(address="auto")
    jax.config.update('jax_platform_name', 'cpu')
    global_config.use_dummy_value_for_benchmarking = True

    cases = build_cases()

    for case in cases:
        benchmark_one_case(case)
