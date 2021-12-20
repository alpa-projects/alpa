import argparse
import time
from datetime import datetime

import numpy as np
import ray

from parax.util import write_tsv, run_cmd, get_num_hosts_and_num_devices, to_str_round
from benchmark.parax.benchmark_moe_3d_one_case import benchmark_one_case
from benchmark.parax.paper_manual_moe_suite import test_moe_suite, paper_moe_suite
from benchmark.parax.paper_auto_moe_suite import test_auto_moe_suite, paper_auto_moe_suite

GB = 1024 ** 3
_ = None

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, S_ = expert_group_size, E = expert_number,
# LD0 = logical_mesh_dimension_0, LD1 = logical_mesh_dimension_1,
# PD0 = physical_mesh_dimension_0, PD1 = physical_mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping, Remat = use_rematerialization
# RS = prefer_reduce_scatter, AP = auto_pipeline

# yapf: disable

default_suite = {
4: [
],

8: [
    #B,  S,    H,    L, #head, V,     E,  S_,            LD0, LD1, PD0, PD1, PP, NB, FM,   Remat, RS,    AP,    _
    (16, 1024, 768,  4, 16,    32000, 16, 8 * 1024 // 2, 1,   4,   1,   4,   2,  1,  False,True,  True,  False, _),
    (16, 1024, 768,  4, 16,    32000, 16, 8 * 1024 // 2, 1,   4,   1,   4,   2,  1,  False,True,  True,  True,  _),
]
}

benchmark_suites = {
    "default": default_suite,
    "test_moe": test_moe_suite,
    "paper_moe": paper_moe_suite,
    "test_auto_moe": test_auto_moe_suite,
    "paper_auto_moe": paper_auto_moe_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=5,
        help="Number of benchmark iteration")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()),
                        default="paper_moe")
    parser.add_argument("--no-separate-process", action='store_false',
                        help="Do not launch separate processes for benchmark."
                             "Errors in a single case will terminate this script.",
                        dest='use_separate_process')
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    print(f"- Use separate process: {args.use_separate_process}")

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
        num_micro_batches, force_data_parallel, use_remat, prefer_reduce_scatter, \
        auto_pipeline, overwrite_global_config_dict = case
        model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads, num_experts, expert_group_size)

        if pipeline_mp_size <= 1 and not auto_pipeline:
            print(f"Skipping the case: {str(case)}, because PP <= 1.")
            continue

        # Run one case
        print(">>> Working on case: {}".format(str(case)))
        result = benchmark_one_case(case, args.niter, num_hosts, num_devices_per_host,
                                    use_separate_process=args.use_separate_process,
                                    disable_tqdm=args.disable_tqdm)
        (parameter_count, mem_allocated, max_mem_allocated, latencies, tflops,
         tflops_ckpt, compilation_times, compute_cost_file_name, forward_stage_layer_ids,
         submesh_shapes, logical_mesh_shapes, autosharding_global_configs) = result

        if not auto_pipeline:
            heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape", "#Microbatch",
                     "Force Mapping", "Remat", "Reduce-scatter", "Mean Time", "Std Time", "#Params", "TFLOPs",
                     "TFLOPs (ckpt)", "Peak Mem"]

            paralell_config = (l_dim0, l_dim1, pipeline_mp_size)
            p_mesh_shape = (p_dim0, p_dim1)
            values = ["MoE", str(model_config), str(paralell_config),
                      str(p_mesh_shape), str(num_micro_batches), str(force_data_parallel), str(use_remat),
                      str(prefer_reduce_scatter), f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                      f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                      f"{max_mem_allocated/GB:5.3f}G" ]
            write_tsv(heads, values, output_name)
        else:
            heads = ["Type", "Model Config", "#GPUs", "#Layers (for Auto-Layer)",
                     "#Microbatch", "Remat", "Reduce-scatter",
                     "Mean Time", "Std Time", "#Params", "TFLOPs",
                     "TFLOPs (ckpt)", "Peak Mem", "Compute Cost File",
                     "Layer->Stage Mapping", "Submesh Shapes",
                     "Logical Mesh Shapes", "Autosharding Global Configs",
                     "overwrite_global_config_dict", "compilation times"]
            values = ["MoE-auto", model_config, num_gpus, pipeline_mp_size,
                      num_micro_batches, use_remat, prefer_reduce_scatter,
                      f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                      f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                      f"{max_mem_allocated/GB:.3f}G", compute_cost_file_name,
                      forward_stage_layer_ids, submesh_shapes,
                      logical_mesh_shapes, autosharding_global_configs,
                      overwrite_global_config_dict, to_str_round(compilation_times, 2)]
            write_tsv(heads, values, output_name)

        time.sleep(0.1)  # for ctrl+c to work
