"""Intra-op parallelism only e2e evaluation."""
import argparse
import time

import numpy as np
import ray

from alpa import DeviceCluster, global_config
from alpa.util import write_tsv, to_str_round, GB
from benchmark_2d_one_case import benchmark_one_case
from suite_paper_manual_gpt import gpt_specs
from suite_paper_manual_moe import moe_specs
from suite_paper_wresnet import wresnet_specs

benchmark_one_case_gpt = (lambda case, niter, num_host, num_devices_per_host:
    benchmark_one_case("gpt", case, niter, num_host, num_devices_per_host,
                       local=False, use_separate_process=True))

benchmark_one_case_moe = (lambda case, niter, num_host, num_devices_per_host:
    benchmark_one_case("moe", case, niter, num_host, num_devices_per_host,
                       local=False, use_separate_process=True))

benchmark_one_case_wresnet = (lambda case, niter, num_host, num_devices_per_host:
    benchmark_one_case("wresnet", case, niter, num_host, num_devices_per_host,
                       local=False, use_separate_process=True))

_ = None

gpt_intra_only = [
    # model,                   LD0, LD1, PD0, PD1, PP, NB,  FM,    Remat, RS,    Other, _
    (1024, *gpt_specs["350M"], 1,   1,   _,   _,   1,  64,  True,  True,  True,  _, _),
    (1024, *gpt_specs["760M"], 2,   1,   _,   _,   1,  64,  True,  True,  True,  _, _),
    (1024, *gpt_specs["1.3B"], 4,   1,   _,   _,   1,  64,  True,  True,  True,  _, _),
    (1024, *gpt_specs["2.6B"], 4,   2,   _,   _,   1,  32,  True,  True,  True,  _, _),
    (1024, *gpt_specs["6.7B"], 2,   8,   _,   _,   1,  64,  True,  True,  True,  _, _),
    (64,   *gpt_specs["15B"],  4,   8,   _,   _,   1,  8,   True,  True,  True,  _, _), # all_reduce_threshold = 1<<20
    (128,  *gpt_specs["15B"],  4,   8,   _,   _,   1,  16,  True,  True,  True,  _, _),
    (256,  *gpt_specs["15B"],  4,   8,   _,   _,   1,  32,  True,  True,  True,  _, _),
    (16,   *gpt_specs["39B"],  2,   32,  _,   _,   1,  8,   True,  True,  True,  _, _),
    (32,   *gpt_specs["39B"],  2,   32,  _,   _,   1,  16,  True,  True,  True,  _, _),
    (64,   *gpt_specs["39B"],  2,   32,  _,   _,   1,  32,  True,  True,  True,  _, _),
]

moe_intra_only = [
    # model,                   S_,   LD0, LD1, PD0, PD1, PP, NB, FM,    Remat, RS,    Other, _
    (1024, *moe_specs["380M"], 2048, 1,   1,   _,   _,   1,  32, False, True,  True,  _, _),
    (1024, *moe_specs["690M"], 2048, 2,   1,   _,   _,   1,  32, False, True,  True,  _, _),
    (1024, *moe_specs["1.3B"], 2048, 4,   1,   _,   _,   1,  16, False, True,  True,  _, _),
    (1024, *moe_specs["2.4B"], 2048, 8,   1,   _,   _,   1,  16, False, True,  True,  _, _),
    (1024, *moe_specs["10B"],  2048, 2,   8,   _,   _,   1,  32, False, True,  True,  _, _),
    (64,   *moe_specs["27B"],  2048, 1,   32,  _,   _,   1,  2,  False, True,  True,  _, _),
    (128,  *moe_specs["27B"],  2048, 1,   32,  _,   _,   1,  4,  False, True,  True,  _, _),
    (256,  *moe_specs["27B"],  2048, 1,   32,  _,   _,   1,  8,  False, True,  True,  _, _),
    (32,   *moe_specs["70B"],  2048, 1,   64,  _,   _,   1,  8,  True,  True,  True,  _, _), # allow_all_to_all = False
    (64,   *moe_specs["70B"],  2048, 1,   64,  _,   _,   1,  16, True,  True,  True,  _, _),
    (128,  *moe_specs["70B"],  2048, 1,   64,  _,   _,   1,  32, True,  True,  True,  _, _),
]

wresnet_intra_only = [
    # model,                       D0, D1, NB, FM,    RS,   Remat, other
    (1536, *wresnet_specs["250M"], 1,  1,  48, False, True, _, _),
    (1536, *wresnet_specs["500M"], 2,  1,  32, False, True, _, _),
    (1536, *wresnet_specs["1B"],   4,  1,  32, False, True, _, _),
    (1536, *wresnet_specs["2B"],   8,  1,  48, False, True, _, _),
    (1536, *wresnet_specs["4B"],   2,  8,  64, False, True, _, _),
    (64,   *wresnet_specs["6.8B"], 4,  8,  2,  False, True, _, _),
    (128,  *wresnet_specs["6.8B"], 4,  8,  4,  False, True, _, _),
    (256,  *wresnet_specs["6.8B"], 4,  8,  8,  False, True, _, _),
    (16,   *wresnet_specs["13B"],  4,  16, 2,  False, True, _, _),
    (32,   *wresnet_specs["13B"],  4,  16, 4,  False, True, _, _),
    (64,   *wresnet_specs["13B"],  4,  16, 8,  False, True, _, _),
]

suites = [
    # GPT
    ("gpt", "alpa.intra_only", gpt_intra_only, benchmark_one_case_gpt),

    # MoE
    ("moe", "alpa.intra_only", moe_intra_only, benchmark_one_case_moe),

    # Wide-ResNet
    ("wresnet", "alpa.intra_only", wresnet_intra_only, benchmark_one_case_wresnet),
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
    cases = build_cases(args.model)
    niter = 3

    for case in cases:
        exp_name, instance, num_hosts, num_devices_per_host, model_name,\
            method, benchmark_func, args = case

        # Benchmark case
        result = benchmark_func(args, niter, num_hosts, num_devices_per_host)
        param_count, ilp_objective, peak_mem, latencies, tflops = result
        if np.mean(latencies) < 0:
            tflops = -1

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
