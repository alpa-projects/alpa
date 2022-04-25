"""The entry point of intra-op parallelism only benchmark."""
import argparse
from datetime import datetime

import numpy as np

from alpa.util import write_tsv, run_cmd, get_num_hosts_and_num_devices, GB

from benchmark_2d_one_case import benchmark_one_case
import suite_manual_gpt
import suite_manual_moe
import suite_wresnet


benchmark_suites = {
    "gpt.tmp": suite_manual_gpt.tmp_suite,
    "gpt.perf_test_fast_2d": suite_manual_gpt.perf_test_fast_2d_suite,

    "moe.tmp": suite_manual_moe.tmp_suite,
    "moe.perf_test_fast_2d": suite_manual_moe.perf_test_fast_2d_suite,

    "wresnet.tmp": suite_wresnet.tmp_suite,
    "wresnet.perf_test_2d": suite_wresnet.perf_test_2d_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()), type=str, required=True)
    parser.add_argument("--niter", type=int, default=5,
                        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int)
    parser.add_argument("--num-devices-per-host", type=int)
    parser.add_argument("--local", action="store_true",
                        help="Run on local GPUs. Do not use ray actors.")
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

    model_type = args.suite.split(".")[0]
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"{model_type}_alpa_{args.exp_name}_{date_str}.tsv"

    # Run all cases
    for benchmark_case in suite:
        if model_type in ["gpt", "bert"]:
            (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
             l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size, num_micro_batches, force_batch_dim_mapping,
             use_remat, prefer_reduce_scatter, pipeline_stage_mode, overwrite_global_config_dict) = benchmark_case
            model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads)
        elif model_type == "moe":
            (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts, expert_group_size,
             l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size,
             num_micro_batches, force_batch_dim_mapping, use_remat, prefer_reduce_scatter,
             auto_pipeline, overwrite_global_config_dict) = benchmark_case
            model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads, num_experts, expert_group_size)
        elif model_type == "wresnet":
            (batch_size, image_size, num_layers, num_channels, width_factor, dtype,
             l_dim0, l_dim1, num_micro_batches, force_batch_dim_mapping,
             prefer_reduce_scatter, use_remat, other) = benchmark_case
            model_config = (batch_size, image_size, num_layers, num_channels, width_factor)
            pipeline_mp_size = 1
        else:
            raise ValueError(f"Invalid model: {model_type}")

        parallel_config = (l_dim0, l_dim1, pipeline_mp_size)

        if pipeline_mp_size > 1:
            print(f"Skipping the case: {str(benchmark_case)}, because PP > 1. "
                  f"Please use `benchmark_gpt_bert_3d.py`.")
            continue

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(model_type, benchmark_case, args.niter,
                                    num_hosts, num_devices_per_host,
                                    args.local, args.use_separate_process)
        param_count, ilp_objective, peak_mem, latencies, tflops = result
        if np.mean(latencies) < 0:
            tflops = -1

        # Log results
        heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape",
                 "#Microbatch", "Force Mapping", "Remat", "Reduce-scatter",
                 "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem", "ILP objective"]
        values = [model_type, model_config, parallel_config, "N/A",
                  num_micro_batches, force_batch_dim_mapping, use_remat, prefer_reduce_scatter,
                  f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{param_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops:.2f}",
                  f"{peak_mem/GB:.3f}G", f"{ilp_objective:.2f}" ]
        write_tsv(heads, values, output_name)
