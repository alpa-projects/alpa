import argparse
import numpy as np
import ray
from datetime import datetime

from benchmark.parax.paper_manual_moe_suite import test_moe_suite, paper_moe_suite
from benchmark.parax.benchmark_moe_3d_one_case import benchmark_one_case
from benchmark.util import run_cmd
from parax.util import (write_tsv)

GB = 1024 ** 3


benchmark_suites = {
    "test_moe": test_moe_suite,
    "paper_moe": paper_moe_suite
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=7,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["paper_moe", "test_moe"], default="paper_moe")
    parser.add_argument("--no-separate-process", action='store_false',
                        help="Do not launch separate processes for benchmark."
                             "Errors in a single case will terminate this script.",
                        dest='use_separate_process')
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()

    print(f"- Use separate process: {args.use_separate_process}")

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
    output_name = f"moe_parax_{args.exp_name}_{date_str}.tsv"
    for case in suite:
        batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts, expert_group_size, \
        l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, \
        num_micro_batches, force_data_parallel, use_remat, tie_word_embeddings, \
        auto_layer, _ = case
        if pipeline_mp_size < 1:
            print(f"Skipping the case: {str(case)}, because PP <= 1. Lianmin will test it.")
            continue

        result = benchmark_one_case(case, args.niter,
                                    use_separate_process=args.use_separate_process)
        parameter_count, mem_allocated, max_mem_allocated, latencies, tflops, tflops_ckpt = result


        heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape", "#Microbatch",
                 "Force DP", "Remat", "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem"]

        model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads, num_experts, expert_group_size)
        paralell_config = (l_dim0, l_dim1, pipeline_mp_size)
        p_mesh_shape = (p_dim0, p_dim1)
        values = ["MoE", str(model_config), str(paralell_config),
                  str(p_mesh_shape), str(num_micro_batches), str(force_data_parallel), str(use_remat),
                  f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                  f"{max_mem_allocated/GB:5.3f}G" ]
        write_tsv(heads, values, output_name)
