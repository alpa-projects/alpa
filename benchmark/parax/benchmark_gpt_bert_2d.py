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
# RS = prefer_reduce_scatter

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    #B,   S,     H     L,   #head,   V,   LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    Auto-pipeline
    (8,  1024,  1024,  4,    32,   51200, 1,   1,   _,   _,    1,   1,  True, True,  False, _),
],

4: [
    #B,   S,     H     L,   #head,   V,   LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    Auto-pipeline
],

8: [
    #B,   S,     H     L,   #head,   V,   LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    Auto-pipeline
    (8,  1024,  1024,  4,    32,   51200, 1,   8,   _,   _,    1,   1,  True, True,  False, _),
    (8,  1024,  1024,  4,    32,   51200, 8,   1,   _,   _,    1,   1,  True, True,  False, _),
],

16: [
    #B,   S,     H     L,   #head,   V,   LD0, LD1, PD0, PD1,  PP,  NB, FM,   Remat, RS,    Auto-pipeline
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
    parser.add_argument("--niter", type=int, default=10,
        help="Number of benchmark iteration")
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()), default="default")
    parser.add_argument("--use-separate-process", action="store_true",
        help="Launch separate processes for benchmark to isolate errors."
              "Errors in a single case will not terminate this script.")
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()
    # Get benchmark suite and run all cases
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        ray.init(address="auto")
        num_gpus = int(ray.cluster_resources()["GPU"])
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
    for case in suite:
        dp, mp, pp = case[6], case[7], case[10]
        if pp > 1:
            print(f"Skipping the case: {str(case)}, because PP > 1. "
                  f"Please use `benchmark_gpt_bert_3d.py`.")
            continue
        print("Working on case: {}".format(str(case)))
        result = benchmark_one_case(args.model, case, args.niter, args.local,
                                    args.use_separate_process)
        param_count, ilp_objective, peak_mem, latencies, tflops = result

        # Log results
        heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape",
                 "#Microbatch", "Force DP", "Remat", "Reduce-scatter",
                 "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem", "ILP objective"]
        parallel_config = (dp, mp, pp)
        values = [args.model, case[:6], parallel_config, "N/A",
                  str(case[11]), str(case[12]), str(case[13]), str(case[14]),
                  f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{param_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops:.2f}",
                  f"{peak_mem/GB:.3f}G", f"{ilp_objective:.2f}" ]
        write_tsv(heads, values, output_name)
