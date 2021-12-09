import time

import numpy as np
import ray

from parax import DeviceCluster, global_config
from parax.util import write_tsv, to_str_round
from benchmark_gpt_bert_2d_one_case import benchmark_one_case as benchmark_one_case_gpt_raw
from benchmark_moe_2d_one_case import benchmark_one_case as benchmark_one_case_moe_raw
from benchmark_wide_resnet_2d_one_case import benchmark_one_case as benchmark_one_case_wresnet_raw

benchmark_one_case_gpt = lambda case, niter, num_host, num_devices_per_host: \
    benchmark_one_case_gpt_raw("gpt", case, niter, num_host, num_devices_per_host,
                               local=False, use_separate_process=True)

benchmark_one_case_moe = lambda case, niter, num_host, num_devices_per_host: \
    benchmark_one_case_moe_raw(case, niter, num_host, num_devices_per_host,
                                   local=False, use_separate_process=True)

benchmark_one_case_wresnet = lambda case, niter, num_host, num_devices_per_host: \
    benchmark_one_case_wresnet_raw(case, niter, num_host, num_devices_per_host,
                                   local=False, use_separate_process=True)


GB = 1 << 30

gpt_1_spec = [
    #B,   S,    H     L, #head,     V,    
    (8,   1024, 2048, 8, 2048//128, 25600),
    (8,   1024, 3072, 8, 3072//128, 25600),
    (8,   1024, 4096, 8, 4096//128, 25600),
    (8,   1024, 5760, 8, 5760//144, 25600),
]

_ = None

gpt_auto_sharding = [
    # model,         LD0, LD1, PD0, PD1,  PP,  NB, FM,    Remat, RS,    Other
    (*gpt_1_spec[0], 1,   1,   _,   _,    _,   1,  True,  True,  True,  _),

    (*gpt_1_spec[1], 2,   1,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[1], 1,   2,   _,   _,    _,   1,  True,  True,  True,  _),

    (*gpt_1_spec[2], 4,   1,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[2], 2,   2,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[2], 1,   4,   _,   _,    _,   1,  True,  True,  True,  _),

    (*gpt_1_spec[3], 8,   1,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[3], 4,   2,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[3], 2,   4,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[3], 1,   8,   _,   _,    _,   1,  True,  True,  True,  _),
]

gpt_data_parallel = [
    # model,         LD0, LD1, PD0, PD1,  PP,  NB, FM,    Remat, RS,    Other
    (*gpt_1_spec[0], 1,   1,   _,   _,    _,   1,  True,  True,  False, _),
    (*gpt_1_spec[1], 2,   1,   _,   _,    _,   1,  True,  True,  False, _),
    (*gpt_1_spec[2], 4,   1,   _,   _,    _,   1,  True,  True,  False, _),
    (*gpt_1_spec[3], 8,   1,   _,   _,    _,   1,  True,  True,  False, _),
]

gpt_zero_2 = [
    # model,         LD0, LD1, PD0, PD1,  PP,  NB, FM,    Remat, RS,    Other
    (*gpt_1_spec[0], 1,   1,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[1], 2,   1,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[2], 4,   1,   _,   _,    _,   1,  True,  True,  True,  _),
    (*gpt_1_spec[3], 8,   1,   _,   _,    _,   1,  True,  True,  True,  _),
]

gpt_zero_3 = [
    # model,         LD0, LD1, PD0, PD1,  PP,  NB, FM,    Remat, RS,    Other
    (*gpt_1_spec[0], 1,   1,   _,   _,    _,   1,  True,  True,  True,  "zero-3"),
    (*gpt_1_spec[1], 2,   1,   _,   _,    _,   1,  True,  True,  True,  "zero-3"),
    (*gpt_1_spec[2], 4,   1,   _,   _,    _,   1,  True,  True,  True,  "zero-3"),
    (*gpt_1_spec[3], 8,   1,   _,   _,    _,   1,  True,  True,  True,  "zero-3"),
]

gpt_heuristic = [
    # model,         LD0, LD1, PD0, PD1,  PP,  NB, FM,    Remat, RS,    Other
    (*gpt_1_spec[0], 1,   1,   _,   _,    _,   1,  True,  True,  True,  "shard-largest"),
    (*gpt_1_spec[1], 2,   1,   _,   _,    _,   1,  False, True,  False, "shard-largest"),
    (*gpt_1_spec[2], 4,   1,   _,   _,    _,   1,  False, True,  False, "shard-largest"),
    (*gpt_1_spec[3], 8,   1,   _,   _,    _,   1,  False, True,  False, "shard-largest"),
]

wresnet1_spec = [
    #B,   I,   L,  C,   W, dtype,  
    (32,  224, 50, 160, 2, "fp32"), 
    (32,  224, 50, 224, 2, "fp32"), 
    (32,  224, 50, 320, 2, "fp32"), 
    (32,  224, 50, 448, 2, "fp32"), 
]

wresnet_auto_sharding = [
    #model,             D0, D1, NB, FM,    RS,   Remat, other
    (*wresnet1_spec[0], 1,  1,  1,  False, True, _,     _),
    (*wresnet1_spec[1], 2,  1,  1,  False, True, _,     _),
    (*wresnet1_spec[2], 4,  1,  1,  False, True, _,     _),
    (*wresnet1_spec[2], 2,  2,  1,  False, True, _,     _),
    (*wresnet1_spec[3], 8,  1,  1,  False, True, _,     _),
    (*wresnet1_spec[3], 4,  2,  1,  False, True, _,     _),
    (*wresnet1_spec[3], 2,  4,  1,  False, True, _,     _),
]

wresnet_data_parallel = [
    #model,             D0, D1, NB, FD,    RS,   Remat, other
    (*wresnet1_spec[0], 1,  1,  1,  True,  False, _,     _),
    (*wresnet1_spec[1], 2,  1,  1,  True,  False, _,     _),
    (*wresnet1_spec[2], 4,  1,  1,  True,  False, _,     _),
    (*wresnet1_spec[3], 8,  1,  1,  True,  False, _,     _),
]

wresnet_zero_2 = [
    #model,             D0, D1, NB, FD,    RS,   Remat, other
    (*wresnet1_spec[0], 1,  1,  1,  True,  True, _,     _),
    (*wresnet1_spec[1], 2,  1,  1,  True,  True, _,     _),
    (*wresnet1_spec[2], 4,  1,  1,  True,  True, _,     _),
    (*wresnet1_spec[3], 8,  1,  1,  True,  True, _,     _),
]

wresnet_zero_3 = [
    #model,             D0, D1, NB, FD,    RS,   Remat, other
    (*wresnet1_spec[0], 1,  1,  1,  True,  True, _,     "zero-3"),
    (*wresnet1_spec[1], 2,  1,  1,  True,  True, _,     "zero-3"),
    (*wresnet1_spec[2], 4,  1,  1,  True,  True, _,     "zero-3"),
    (*wresnet1_spec[3], 8,  1,  1,  True,  True, _,     "zero-3"),
]

wresnet_heuristic = [
    #model,             D0, D1, NB, FD,    RS,   Remat, other
    (*wresnet1_spec[0], 1,  1,  1,  True,  True, _,     "shard-largest"),
    (*wresnet1_spec[1], 2,  1,  1,  True,  True, _,     "shard-largest"),
    (*wresnet1_spec[2], 4,  1,  1,  True,  True, _,     "shard-largest"),
    (*wresnet1_spec[3], 8,  1,  1,  True,  True, _,     "shard-largest"),
]

moe1_spec = [
    #B, S,    H     L, #head,    V,     E,  S_,
    (8, 1024, 1024, 8, 1024//32, 51200, 8,  1024),
    (8, 1024, 1280, 8, 1280//32, 51200, 16, 1024),
    (8, 1024, 1536, 8, 1536//64, 51200, 16, 1024),
    (8, 1024, 1536, 8, 1536//64, 51200, 32, 1024),
]

moe_auto_sharding = [
    #model,         LD0, LD1, PD0, PD1, PP, NB, FM,    Remat, RS,    Other, _
    (*moe1_spec[0], 1,   1,   _,   _,   _,  1,  False, True,  True,  _,  _),
    (*moe1_spec[1], 2,   1,   _,   _,   _,  1,  False, True,  True,  _,  _),
    (*moe1_spec[2], 4,   1,   _,   _,   _,  1,  False, True,  True,  _,  _),
    (*moe1_spec[2], 2,   2,   _,   _,   _,  1,  False, True,  True,  _,  _),
    (*moe1_spec[3], 8,   1,   _,   _,   _,  1,  False, True,  True,  _,  _),
    (*moe1_spec[3], 4,   2,   _,   _,   _,  1,  False, True,  True,  _,  _),
    (*moe1_spec[3], 2,   4,   _,   _,   _,  1,  False, True,  True,  _,  _),
]

moe_data_parallel = [
    #model,         LD0, LD1, PD0, PD1, PP, NB, FM,    Remat, RS,    Other, _
    (*moe1_spec[0], 1,   1,   _,   _,   _,  1,  True,  True,  False, _,  _),
    (*moe1_spec[1], 2,   1,   _,   _,   _,  1,  True,  True,  False, _,  _),
    (*moe1_spec[2], 4,   1,   _,   _,   _,  1,  True,  True,  False, _,  _),
    (*moe1_spec[3], 8,   1,   _,   _,   _,  1,  True,  True,  False, _,  _),
]

moe_zero_2 = [
    #model,         LD0, LD1, PD0, PD1, PP, NB, FM,    Remat, RS,    Other, _
    (*moe1_spec[0], 1,   1,   _,   _,   _,  1,  True,  True,  True,  _,  _),
    (*moe1_spec[1], 2,   1,   _,   _,   _,  1,  True,  True,  True,  _,  _),
    (*moe1_spec[2], 4,   1,   _,   _,   _,  1,  True,  True,  True,  _,  _),
    (*moe1_spec[3], 8,   1,   _,   _,   _,  1,  True,  True,  True,  _,  _),
]

moe_zero_3 = [
    #model,         LD0, LD1, PD0, PD1, PP, NB, FM,    Remat, RS,    Other, _
    (*moe1_spec[0], 1,   1,   _,   _,   _,  1,  True,  True,  True,  "zero-3", _),
    (*moe1_spec[1], 2,   1,   _,   _,   _,  1,  True,  True,  True,  "zero-3", _),
    (*moe1_spec[2], 4,   1,   _,   _,   _,  1,  True,  True,  True,  "zero-3", _),
    (*moe1_spec[3], 8,   1,   _,   _,   _,  1,  True,  True,  True,  "zero-3", _),
]

moe_heuristic = [
    #model,         LD0, LD1, PD0, PD1, PP, NB, FM,    Remat, RS,    Other, _
    (*moe1_spec[0], 1,   1,   _,   _,   _,  1,  True,  True,  True,  "shard-largest", _),
    (*moe1_spec[1], 2,   1,   _,   _,   _,  1,  True,  True,  True,  "shard-largest", _),
    (*moe1_spec[2], 4,   1,   _,   _,   _,  1,  True,  True,  True,  "shard-largest", _),
    (*moe1_spec[3], 8,   1,   _,   _,   _,  1,  True,  True,  True,  "shard-largest", _),
]

suites = [
    # GPT
    ("GPT-1", "parax.auto_sharding", gpt_auto_sharding, benchmark_one_case_gpt),
    ("GPT-1", "parax.data_parallel", gpt_data_parallel, benchmark_one_case_gpt),
    ("GPT-1", "parax.zero_2", gpt_zero_2, benchmark_one_case_gpt),
    ("GPT-1", "parax.zero_3", gpt_zero_3, benchmark_one_case_gpt),
    ("GPT-1", "parax.heuristic", gpt_heuristic, benchmark_one_case_gpt),

    # W-resnet
    ("W-ResNet-1", "parax.auto_sharding", wresnet_auto_sharding, benchmark_one_case_wresnet),
    ("W-ResNet-1", "parax.data_parallel", wresnet_data_parallel, benchmark_one_case_wresnet),
    ("W-ResNet-1", "parax.zero_2", wresnet_zero_2, benchmark_one_case_wresnet),
    ("W-ResNet-1", "parax.zero_3", wresnet_zero_3, benchmark_one_case_wresnet),
    ("W-ResNet-1", "parax.heuristic", wresnet_heuristic, benchmark_one_case_wresnet),

    # MoE
    ("MoE-1", "parax.auto_sharding", moe_auto_sharding, benchmark_one_case_moe),
    ("MoE-1", "parax.data_parallel", moe_data_parallel, benchmark_one_case_moe),
    ("MoE-1", "parax.zero_2", moe_zero_2, benchmark_one_case_moe),
    ("MoE-1", "parax.zero_3", moe_zero_3, benchmark_one_case_moe),
    ("MoE-1", "parax.heuristic", moe_heuristic, benchmark_one_case_moe),
]


def build_cases():
    instance = "p3.16"
    exp_name = "intra-op-ablation"

    cases = []
    for suite in suites:
        model_name, method, args_list, benchmark_func = suite
        for i, args in enumerate(args_list):
            if "MoE" in model_name:
                num_devices = args[8] * args[9]
            else: # GPT and W-ResNet
                num_devices = args[6] * args[7]
            num_hosts = ((num_devices + 7) // 8)
            num_devices_per_host = min(num_devices, 8)
            cases.append((exp_name, instance, num_hosts, num_devices_per_host,
                          model_name, method, benchmark_func, args))

    return cases


if __name__ == "__main__":
    cases = build_cases()

    for case in cases:
        exp_name, instance, num_hosts, num_devices_per_host, model_name,\
            method, benchmark_func, args = case

        # Benchmark case
        result = benchmark_func(args, 5, num_hosts, num_devices_per_host)
        param_count, ilp_objective, peak_mem, latencies, tflops = result
        value_dict = {
            "param_count": param_count / 1e9,
            "peak_mem": peak_mem / GB,
            "latencies": latencies,
            "tflops": tflops,
        }

        # Log results
        heads = ["Exp", "Instance", "num_hosts", "num_devices_per_host", "model_name", 
                 "method", "value", "tstamp"]
        values = [exp_name, instance, num_hosts, num_devices_per_host,
                  model_name, method, to_str_round(value_dict, 4),
                  int(time.time())]
        write_tsv(heads, values, f"results_intra_ablation.tsv")
