import argparse
from datetime import datetime

import numpy as np
import ray

from benchmark.parax.benchmark_moe_2d_one_case import benchmark_one_case
from benchmark.parax.paper_manual_moe_suite import paper_moe_suite, test_moe_suite
from parax.util import (run_cmd, list_gpu_info, write_tsv)

GB = 1024 ** 3


benchmark_suites = {
    "test_moe": test_moe_suite,
    "paper_moe": paper_moe_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--niter", type=int, default=7,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["default", "paper_moe", "test_moe"], default="default")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--use-separate-process", action="store_true",
        help="Launch separate processes for benchmark to isolate errors."
             "Errors in a single case will not terminate this script.")
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()

    # Set global environments
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        ray.init(address="auto")
        num_gpus = int(ray.cluster_resources()["GPU"])

    # Get benchmark suite and run all cases
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

    for case in suite:
        batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts, expert_group_size, \
        l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, \
        num_micro_batches, force_data_parallel, use_remat, reduce_scatter, \
        auto_layer, _ = case
        if pipeline_mp_size > 1:
            print(f"Skipping the case: {str(case)}, because PP > 1. "
                  f"Please use `benchmark_gpt_bert_3d.py`.")
            continue
        print(">>> Working on case: {}".format(str(case)))
        result = benchmark_one_case(case, args.niter, args.local,
                                    args.use_separate_process)
        parameter_count, ilp_objective, peak_mem, latencies, tflops = result

        heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape",
                 "#Microbatch", "Force Mapping", "Remat", "Reduce-scatter",
                 "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem", "ILP objective"]

        model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads, num_experts, expert_group_size)
        paralell_config = (l_dim0, l_dim1, pipeline_mp_size)
        p_mesh_shape = (p_dim0, p_dim1)

        values = ["MOE", str(model_config), str(paralell_config), "N/A",
                  num_micro_batches, force_data_parallel, use_remat, reduce_scatter,
                  f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops:.2f}",
                  f"{peak_mem/GB:.3f}G", ilp_objective]
        write_tsv(heads, values, output_name)
