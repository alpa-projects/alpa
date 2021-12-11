import argparse
from datetime import datetime

import numpy as np
import ray

from parax.util import run_cmd, write_tsv, get_num_hosts_and_num_devices
from benchmark.parax.benchmark_moe_2d_one_case import benchmark_one_case
from benchmark.parax.paper_manual_moe_suite import paper_moe_suite, test_moe_suite

GB = 1024 ** 3

_ = None

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, S_ = expert_group_size, E = expert_number,
# LD0 = logical_mesh_dimension_0, LD1 = logical_mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping, Remat = use_rematerialization
# RS = prefer_reduce_scatter, AP = auto_pipeline

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    #B, S,    H     L, #head, V,     E,  S_,   LD0, LD1, _, _,  PP,  NB, FM,    Remat, RS,    _, _
    (8, 1024, 1024, 8, 32,    25600, 8,  1024, 1,   1,   _, _,  1,   1,  True,  True,  False, _, _),
],

8: [
    #B, S,    H     L, #head, V,     E,  S_,   LD0, LD1, _, _,  PP,  NB, FM,    Remat, RS,    _, _
    #(8, 1024, 1024, 4, 32,    25600, 16, 1024, 8,   1,   _, _,  1,   1,  False, True,  False, _, _),

    (8, 1024, 1024, 4, 32,    25600, 16, 1024, 2,   4,   _, _,  1,   1,  False, True,  False, _, _),
],

}

benchmark_suites = {
    "default": default_benchmark_suite,
    "test_moe": test_moe_suite,
    "paper_moe": paper_moe_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    # Get the benchmark suite
    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)
    num_gpus = num_hosts * num_devices_per_host

    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None
    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()
    run_cmd("mkdir -p tmp")

    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"moe_parax_{args.exp_name}_{date_str}.tsv"

    # Run all cases
    for case in suite:
        batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts, expert_group_size, \
        l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, \
        num_micro_batches, force_batch_dim_mapping, use_remat, reduce_scatter, \
        _, _ = case

        # Run one case
        if pipeline_mp_size > 1:
            print(f"Skipping the case: {str(case)}, because PP > 1. "
                  f"Please use `benchmark_gpt_bert_3d.py`.")
            continue
        print(">>> Working on case: {}".format(str(case)))
        result = benchmark_one_case(case, args.niter, num_hosts, 
                                    num_devices_per_host, args.local,
                                    args.use_separate_process)
        parameter_count, ilp_objective, peak_mem, latencies, tflops = result

        # Log results
        heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape",
                 "#Microbatch", "Force Mapping", "Remat", "Reduce-scatter",
                 "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem", "ILP objective"]
        model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads, num_experts, expert_group_size)
        paralell_config = (l_dim0, l_dim1, pipeline_mp_size)
        p_mesh_shape = (p_dim0, p_dim1)
        values = ["MOE", str(model_config), str(paralell_config), "N/A",
                  num_micro_batches, force_batch_dim_mapping, use_remat, reduce_scatter,
                  f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops:.2f}",
                  f"{peak_mem/GB:.3f}G", ilp_objective]
        write_tsv(heads, values, output_name)
