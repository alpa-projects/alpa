"""The entry point of intra-op + inter-op parallelism benchmark."""
import argparse
from datetime import datetime
import time

import numpy as np

from parax.util import write_tsv, run_cmd, get_num_hosts_and_num_devices, to_str_round, GB

from benchmark_3d_one_case import benchmark_one_case
from benchmark_3d_one_case_wresnet import paper_auto_wresnet_suite
from paper_manual_gpt_suite import fast_test_gpt_suite, test_gpt_suite, paper_gpt_suite
from paper_manual_moe_suite import fast_test_moe_suite, test_moe_suite, paper_moe_suite
from paper_auto_gpt_suite import test_auto_gpt_suite, paper_auto_gpt_suite, result_auto_gpt_suite
from paper_auto_moe_suite import test_auto_moe_suite, paper_auto_moe_suite


benchmark_suites = {
    "gpt.fast_test": fast_test_gpt_suite,
    "gpt.test": test_gpt_suite,
    "gpt.paper": paper_gpt_suite,
    "gpt.test_auto": test_auto_gpt_suite,
    "gpt.paper_auto": paper_auto_gpt_suite,
    "gpt.result_auto": result_auto_gpt_suite,

    "moe.fast_test": fast_test_moe_suite,
    "moe.test": test_moe_suite,
    "moe.paper": paper_moe_suite,
    "moe.test_auto": test_auto_moe_suite,
    "moe.paper_auto": paper_auto_moe_suite,

    "wresnet.paper_auto": paper_auto_wresnet_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()), type=str, required=True)
    parser.add_argument("--niter", type=int, default=5,
        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--no-separate-process", action='store_false',
                        help="Do not launch separate processes for benchmark."
                             "Errors in a single case will terminate this script.",
                        dest='use_separate_process')
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
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
    output_name = f"{model_type}_parax_{args.exp_name}_{date_str}.tsv"

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
            pipeline_stage_mode = "auto_gpipe" if auto_pipeline else "uniform_layer_gpipe"
        elif model_type == "wresnet":
            (batch_size, image_size, num_layers, num_channels, width_factor, dtype,
             num_micro_batches, force_batch_dim_mapping,
             prefer_reduce_scatter, use_remat, logical_mesh_search_space) = benchmark_case
            model_config = (batch_size, image_size, num_layers, num_channels, width_factor)
            pipeline_stage_mode = "auto_gpipe"
            pipeline_mp_size = 1
        else:
            raise ValueError(f"Invalid model: {model_type}")

        if pipeline_mp_size <= 1 and pipeline_stage_mode == "uniform_layer_gpipe":
            print(f"Skip the case: {str(benchmark_case)}, because PP <= 1. "
                  f"Please use `benchmark_2d.py` "
                  f"since 3d runtime will have a small overhead.")
            continue

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(model_type, benchmark_case, args.niter,
                                    num_hosts, num_devices_per_host,
                                    use_separate_process=args.use_separate_process,
                                    disable_tqdm=args.disable_tqdm)
        (parameter_count, mem_allocated, max_mem_allocated, latencies, tflops,
         tflops_ckpt, compilation_times, compute_cost_file_name, forward_stage_layer_ids,
         submesh_shapes, logical_mesh_shapes, autosharding_option_dicts) = result

        if pipeline_stage_mode == "uniform_layer_gpipe":
            heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape",
                     "#Microbatch", "Force Mapping", "Remat", "Reduce-scatter",
                     "Mean Time", "Std Time", "#Params", "TFLOPs",
                     "TFLOPs (ckpt)", "Peak Mem", "overwrite_global_config_dict"]
            parallel_config = (l_dim0, l_dim1, pipeline_mp_size)
            values = [model_type, model_config, parallel_config, (p_dim0, p_dim1),
                      num_micro_batches, force_batch_dim_mapping, use_remat, prefer_reduce_scatter,
                      f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                      f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                      f"{max_mem_allocated/GB:.3f}G", overwrite_global_config_dict]
            write_tsv(heads, values, output_name)
        else:
            heads = ["Type", "Model Config", "#GPUs", "#Layers (for Auto-Layer)",
                     "#Microbatch", "Remat", "Reduce-scatter",
                     "Mean Time", "Std Time", "#Params", "TFLOPs",
                     "TFLOPs (ckpt)", "Peak Mem", "Compute Cost File",
                     "Layer->Stage Mapping", "Submesh Shapes",
                     "Logical Mesh Shapes", "Autosharding Global Configs",
                     "overwrite_global_config_dict", "compilation times"]
            values = [model_type + "-" + pipeline_stage_mode, model_config, num_gpus, pipeline_mp_size,
                      num_micro_batches, use_remat, prefer_reduce_scatter,
                      f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                      f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                      f"{max_mem_allocated/GB:.3f}G", compute_cost_file_name,
                      forward_stage_layer_ids, submesh_shapes,
                      logical_mesh_shapes, autosharding_option_dicts,
                      overwrite_global_config_dict, to_str_round(compilation_times, 2)]
            write_tsv(heads, values, output_name)

        time.sleep(0.1)  # for ctrl+c to work
