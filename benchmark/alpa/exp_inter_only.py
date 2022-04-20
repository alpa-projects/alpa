"""Inter-op parallelism only e2e evaluation."""
import argparse
import time

import numpy as np
import ray

from alpa import DeviceCluster, global_config
from alpa.util import write_tsv, to_str_round
from benchmark_3d_one_case import benchmark_one_case
from suite_paper_manual_gpt import gpt_specs
from suite_paper_manual_moe import moe_specs

benchmark_one_case_gpt = (lambda case, niter, num_host, num_devices_per_host:
    benchmark_one_case("gpt", case, niter, num_host, num_devices_per_host,
                       use_separate_process=True))

benchmark_one_case_moe = (lambda case, niter, num_host, num_devices_per_host:
    benchmark_one_case("moe", case, niter, num_host, num_devices_per_host,
                       use_separate_process=True))

benchmark_one_case_wresnet = (lambda case, niter, num_host, num_devices_per_host:
    benchmark_one_case("wresnet", case, niter, num_host, num_devices_per_host,
                       use_separate_process=True))

GB = 1 << 30
_ = None

gpt_inter_only = [
    # model,                   LD0, LD1, PD0, PD1, PP, NB,   FM,    Remat, RS,    Stage, _
    (1024, *gpt_specs["350M"], 1,   1,   1,   1,   1,  64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["350M"], 1,   1,   1,   1,   1,  128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["350M"], 1,   1,   1,   1,   1,  256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["760M"], 1,   1,   1,   1,   2,  64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["760M"], 1,   1,   1,   1,   2,  128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["760M"], 1,   1,   1,   1,   2,  256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["1.3B"], 1,   1,   1,   1,   4,  64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["1.3B"], 1,   1,   1,   1,   4,  128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["1.3B"], 1,   1,   1,   1,   4,  256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["1.3B"], 1,   1,   1,   1,   4,  512,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["2.6B"], 1,   1,   1,   1,   8,  64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["2.6B"], 1,   1,   1,   1,   8,  128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["2.6B"], 1,   1,   1,   1,   8,  256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["2.6B"], 1,   1,   1,   1,   8,  512,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["2.6B"], 1,   1,   1,   1,   8,  1024, True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["6.7B"], 1,   1,   1,   1,   16, 128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["6.7B"], 1,   1,   1,   1,   16, 256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["6.7B"], 1,   1,   1,   1,   16, 512,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["6.7B"], 1,   1,   1,   1,   16, 1024, True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["15B"],  1,   1,   1,   1,   32, 128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["15B"],  1,   1,   1,   1,   32, 256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["15B"],  1,   1,   1,   1,   32, 512,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["15B"],  1,   1,   1,   1,   32, 1024, True,  True,  True,  "uniform_layer_gpipe", _),
    #LAYER_HEAVY_OP_LOWER_BOUND = 1
    #DEFAULT_EPS = 0.4
    #DEFAULT_COST_CRITERIA = "input_memory"
    (128,  *gpt_specs["39B"],  1,   1,   1,   1,   64, 128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (512,  *gpt_specs["39B"],  1,   1,   1,   1,   64, 512,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *gpt_specs["39B"],  1,   1,   1,   1,   64, 1024, True,  True,  True,  "uniform_layer_gpipe", _),
]

moe_inter_only = [
    # model,                   S_,   LD0, LD1, PD0, PD1, PP, NB,   FM,    Remat, RS,    AP, _
    (1024, *moe_specs["380M"], 2048, 1,   1,   1,   1,   1,  64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["380M"], 2048, 1,   1,   1,   1,   1,  128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["690M"], 2048, 1,   1,   1,   1,   2,  64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["690M"], 2048, 1,   1,   1,   1,   2,  128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["690M"], 2048, 1,   1,   1,   1,   2,  256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["1.3B"], 2048, 1,   1,   1,   1,   4,  64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["1.3B"], 2048, 1,   1,   1,   1,   4,  128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["1.3B"], 2048, 1,   1,   1,   1,   4,  256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["1.3B"], 2048, 1,   1,   1,   1,   4,  512,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["2.4B"], 2048, 1,   1,   1,   1,   8,  64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["2.4B"], 2048, 1,   1,   1,   1,   8,  128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["2.4B"], 2048, 1,   1,   1,   1,   8,  256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["2.4B"], 2048, 1,   1,   1,   1,   8,  512,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["2.4B"], 2048, 1,   1,   1,   1,   8,  1024, True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["10B"],  2048, 1,   1,   1,   1,   16, 64,   True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["10B"],  2048, 1,   1,   1,   1,   16, 128,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["10B"],  2048, 1,   1,   1,   1,   16, 256,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["10B"],  2048, 1,   1,   1,   1,   16, 512,  True,  True,  True,  "uniform_layer_gpipe", _),
    (1024, *moe_specs["10B"],  2048, 1,   1,   1,   1,   16, 1024, True,  True,  True,  "uniform_layer_gpipe", _),
    #(1024, *moe_specs["27B"],  2048, 1,   1,   1,   1,   32, 512,  True,  True,  True,  "uniform_layer_gpipe", _), # eps = 6, LAYER_HEAVY_OP_LOWER_BOUND = 1
    #(1024, *moe_specs["27B"],  2048, 1,   1,   1,   1,   32, 1024, True,  True,  True,  "uniform_layer_gpipe", _),
    #(1024, *moe_specs["70B"],  2048, 1,   1,   1,   1,   64, 512,  True,  True,  True,  "uniform_layer_gpipe", _),
    #(1024, *moe_specs["70B"],  2048, 1,   1,   1,   1,   64, 1024, True,  True,  True,  "uniform_layer_gpipe", _),
]

wresnet_inter_only = [
    # model,                       D0, D1, NB, FM,    RS,   Remat, other
]

suites = [
    # GPT
    ("gpt", "alpa.inter_only", gpt_inter_only, benchmark_one_case_gpt),

    # MoE
    ("moe", "alpa.inter_only", moe_inter_only, benchmark_one_case_moe),

    # Wide-ResNet
    ("wresnet", "alpa.inter_only", wresnet_inter_only, benchmark_one_case_wresnet),
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
                num_devices = args[12]
            else: # GPT and W-ResNet
                num_devices = args[10]
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
        (parameter_count, mem_allocated, max_mem_allocated, latencies, tflops,
         tflops_ckpt, compilation_times, compute_cost_file_name, forward_stage_layer_ids,
         submesh_shapes, logical_mesh_shapes, autosharding_option_dicts) = result

        value_dict = {
            "param_count": parameter_count / 1e9,
            "peak_mem": max_mem_allocated / GB,
            "latencies": latencies,
            "tflops": tflops_ckpt if tflops_ckpt > 0.0 else -1.0,
        }

        # Log results
        heads = ["Exp", "Instance", "num_hosts", "num_devices_per_host", "model_name", 
                 "method", "value", "tstamp"]
        values = [exp_name, instance, num_hosts, num_devices_per_host,
                  model_name, method, to_str_round(value_dict, 4),
                  int(time.time())]
        write_tsv(heads, values, f"results_inter_only.tsv")
