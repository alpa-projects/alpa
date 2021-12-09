import argparse
from datetime import datetime

import ray

import numpy as np

from parax.util import list_gpu_info, write_tsv, run_cmd

from benchmark_gpt_bert_2d_one_case import benchmark_one_case
from benchmark.parax.paper_manual_gpt_suite import paper_gpt_suite, test_gpt_suite
from benchmark.parax.paper_auto_gpt_suite import paper_auto_gpt_suite

GB = 1 << 30
_ = None

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, LD0 = logical_mesh_dimension_0, LD1 = logical_mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping, Remat = use_rematerialization
# RS = prefer_reduce_scatter, AP = auto_pipeline

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    #B,   S,     H     L,   #head,   V,   LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    AP
    (8,  1024,  1024,  4,    32,   51200, 1,   1,   _,   _,    1,   1,  True, True,  False, _),
],

4: [
    #B,   S,     H     L,   #head,   V,   LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    AP
],

8: [
    #B,   S,     H     L,   #head,   V,   LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    AP
    (8,  1024,  1024,  4,    32,   51200, 1,   8,   _,   _,    1,   1,  True, True,  False, _),
],

16: [
    #B,   S,     H     L,   #head,   V,   LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    AP
]
}

benchmark_suites = {
    "default": default_benchmark_suite,
    "test_gpt": test_gpt_suite,
    "paper_gpt": paper_gpt_suite,
    "paper_auto_gpt": paper_auto_gpt_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=5,
        help="Number of benchmark iteration")
    parser.add_argument("--num-hosts", type=int)
    parser.add_argument("--num-devices-per-host", type=int)
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()), default="default")
    parser.add_argument("--use-separate-process", action="store_true",
        help="Launch separate processes for benchmark to isolate errors."
              "Errors in a single case will not terminate this script.")
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()

    # Get the number of devices
    if args.num_hosts is not None or args.num_devices_per_host is not None:
        assert args.num_hosts is not None and args.num_devices_per_host is not None
        num_hosts, num_devices_per_host = args.num_hosts, args.num_devices_per_host
    else:
        if args.local:
            num_hosts = 1
            num_devices_per_host = list_gpu_info().count("UUID")
        else:
            ray.init(address="auto")
            num_hosts = len(ray.nodes())
            num_devices_per_host = int(ray.cluster_resources()["GPU"]) // num_hosts
    num_gpus = num_hosts * num_devices_per_host

    # Get the benchmark suite
    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None
    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()
    run_cmd("mkdir -p tmp")

    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"{args.model}_parax_{args.exp_name}_{date_str}.tsv"

    # Run all cases
    for benchmark_case in suite:
        (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
         l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, num_micro_batches, force_batch_dim_mapping,
         use_remat, prefer_reduce_scatter, auto_pipeline) = benchmark_case
        model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads)
        parallel_config = (l_dim0, l_dim1, pipeline_mp_size)

        # Run one case
        if pipeline_mp_size > 1:
            print(f"Skipping the case: {str(benchmark_case)}, because PP > 1. "
                  f"Please use `benchmark_gpt_bert_3d.py`.")
            continue
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(args.model, benchmark_case, args.niter,
                                    num_hosts, num_devices_per_host,
                                    args.local, args.use_separate_process)
        param_count, ilp_objective, peak_mem, latencies, tflops = result

        # Log results
        heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape",
                 "#Microbatch", "Force Mapping", "Remat", "Reduce-scatter",
                 "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem", "ILP objective"]
        values = [args.model, model_config, parallel_config, "N/A",
                  num_micro_batches, force_batch_dim_mapping, use_remat, prefer_reduce_scatter,
                  f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{param_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops:.2f}",
                  f"{peak_mem/GB:.3f}G", f"{ilp_objective:.2f}" ]
        write_tsv(heads, values, output_name)
