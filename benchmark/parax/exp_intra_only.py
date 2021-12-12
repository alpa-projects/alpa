"""Intra operator ablation study"""
import argparse
import time

import numpy as np
import ray

from parax import DeviceCluster, global_config
from parax.util import write_tsv, to_str_round
from benchmark_gpt_bert_2d_one_case import benchmark_one_case as benchmark_one_case_gpt_raw
from benchmark_moe_2d_one_case import benchmark_one_case as benchmark_one_case_moe_raw
from benchmark_wresnet_2d_one_case import benchmark_one_case as benchmark_one_case_wresnet_raw
from paper_manual_gpt_suite import gpt_specs
from paper_manual_moe_suite import moe_specs

benchmark_one_case_gpt = lambda case, niter, num_host, num_devices_per_host: \
    benchmark_one_case_gpt_raw("gpt", case, niter, num_host, num_devices_per_host,
                               local=False, use_separate_process=True)

benchmark_one_case_moe = lambda case, niter, num_host, num_devices_per_host: \
    benchmark_one_case_moe_raw(case, niter, num_host, num_devices_per_host,
                                   local=False, use_separate_process=True)

benchmark_one_case_wresnet = lambda case, niter, num_host, num_devices_per_host: \
    benchmark_one_case_wresnet_raw(case, niter, num_host, num_devices_per_host,
                                   local=False, use_separate_process=True)

wresnet_specs = {
    #    I,   L,  C,   W,  dtype,  
"250M": (224, 50, 160, 2,  "fp32"), 
"500M": (224, 50, 224, 2,  "fp32"), 
"1B":   (224, 50, 320, 2,  "fp32"), 
"2B":   (224, 50, 448, 2,  "fp32"), 
"4B":   (224, 50, 640, 2,  "fp32"), 
"6.8B": (224, 50, 320, 16, "fp32"), 
}

GB = 1 << 30
_ = None

gpt_intra_only = [
    # model,                   LD0, LD1, PD0, PD1, PP, NB,  FM,    Remat, RS,    Other, _
    (1024, *gpt_specs["350M"], 1,   1,   _,   _,   1,  64,  True,  True,  True,  _,     _),
    (1024, *gpt_specs["760M"], 2,   1,   _,   _,   1,  64,  True,  True,  True,  _,     _),
    (1024, *gpt_specs["1.3B"], 4,   1,   _,   _,   1,  64,  True,  True,  True,  _,     _),
    (1024, *gpt_specs["2.7B"], 4,   2,   _,   _,   1,  32,  True,  True,  True,  _,     _),
    (1024, *gpt_specs["6.7B"], 2,   8,   _,   _,   1,  64,  True,  True,  True,  _,     _),
    (1024, *gpt_specs["15B"],  4,   8,  _,   _,   1,  128,  True,  True,  True,  _, _), # reduce_scatter_grad_acc_friendly = False, ALLREDUCE_THRESHOLD = 1 << 20
]

moe_intra_only = [
    # model,                   S_,   LD0, LD1, PD0, PD1, PP, NB, FM,    Remat, RS,    Other, _
    #(1024, *moe_specs["380M"], 2048, 1,   1,   _,   _,   1,  32, False, True,  True,  _,     _),
    #(1024, *moe_specs["690M"], 2048, 2,   1,   _,   _,   1,  32, False, True,  True,  _,     _),
    #(1024, *moe_specs["1.3B"], 2048, 4,   1,   _,   _,   1,  16, False, True,  True,  _,     _),
    #(1024, *moe_specs["2.4B"], 2048, 8,   1,   _,   _,   1,  16, False, True,  True,  _,     _),
    #(1024, *moe_specs["10B"], 2048, 2,   8,   _,   _,   1,  32, False, True,  True,  _,     _),
]

wresnet_intra_only = [
    # model,                       D0, D1, NB, FM,    RS,   Remat, other
    (1536, *wresnet_specs["250M"], 1,  1,  48, False, True, _, _),
    (1536, *wresnet_specs["500M"], 2,  1,  32, False, True, _, _),
    (1536, *wresnet_specs["1B"],   4,  1,  32, False, True, _, _),
    (1536, *wresnet_specs["2B"],   8,  1,  48, False, True, _, _), # MEM_FRACTION = 0.85
    (1536, *wresnet_specs["4B"],   2,  8,  64, False, True, _, _), # MEM_FRACTION = 0.85
    (64,   *wresnet_specs["6.8B"], 4,  8,  2,  False, True, _, _), # MEM_FRACTION = 0.85
    (128,  *wresnet_specs["6.8B"], 4,  8,  4,  False, True, _, _), # MEM_FRACTION = 0.85
    (256,  *wresnet_specs["6.8B"], 4,  8,  8,  False, True, _, _), # MEM_FRACTION = 0.85
    (1536, *wresnet_specs["6.8B"], 4,  8,  8,  False, True, _, _), # projected
]

suites = [
    # GPT
    ("gpt", "parax.intra_only", gpt_intra_only, benchmark_one_case_gpt),

    # MoE
    ("moe", "parax.intra_only", moe_intra_only, benchmark_one_case_moe),

    # Wide-ResNet
    ("wresnet", "parax.intra_only", wresnet_intra_only, benchmark_one_case_wresnet),
]


def build_cases(wanted_model):
    instance = "p3.16"
    exp_name = "e2e"

    cases = []
    for suite in suites:
        model_name, method, args_list, benchmark_func = suite

        if wanted_model is not None and model_name != wanted_model:
            continue

        for i, args in enumerate(args_list):
            if "moe" in model_name:
                num_devices = args[8] * args[9]
            else: # GPT and W-ResNet
                num_devices = args[6] * args[7]
            num_hosts = ((num_devices + 7) // 8)
            num_devices_per_host = min(num_devices, 8)
            cases.append((exp_name, instance, num_hosts, num_devices_per_host,
                          model_name, method, benchmark_func, args))

    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    niter = 4
    cases = build_cases(args.model)

    for case in cases:
        exp_name, instance, num_hosts, num_devices_per_host, model_name,\
            method, benchmark_func, args = case

        # Benchmark case
        result = benchmark_func(args, niter, num_hosts, num_devices_per_host)
        param_count, ilp_objective, peak_mem, latencies, tflops = result
        value_dict = {
            "param_count": param_count / 1e9,
            "peak_mem": peak_mem / GB,
            "latencies": latencies,
            "tflops": tflops if tflops > 0.0 else -1.0,
        }

        # Log results
        heads = ["Exp", "Instance", "num_hosts", "num_devices_per_host", "model_name", 
                 "method", "value", "tstamp"]
        values = [exp_name, instance, num_hosts, num_devices_per_host,
                  model_name, method, to_str_round(value_dict, 4),
                  int(time.time())]
        write_tsv(heads, values, f"results_intra_only.tsv")
