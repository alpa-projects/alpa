import argparse
from datetime import datetime
import time

import numpy as np
import ray

from parax.util import write_tsv, run_cmd
from benchmark.parax.benchmark_gpt_bert_3d_one_case import benchmark_one_case
from benchmark.parax.paper_manual_gpt_suite import paper_gpt_suite, test_gpt_suite
from benchmark.parax.paper_auto_gpt_suite import paper_auto_gpt_suite, test_auto_gpt_suite

GB = 1024 ** 3

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, LD0 = logical_mesh_dimension_0, LD1 = logical_mesh_dimension_1,
# PD0 = physical_mesh_dimension_0, PD1 = physical_mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping, Remat = use_rematerialization
# RS = prefer_reduce_scatter, AP = auto-pipeline

# yapf: disable

default_suite = {
4: [
    #B,   S,     H     L,   #head,   V,      LD0, LD1, PD0, PD1, PP, NB,  FM,    Remat, RS,    AP
    (32,   1024,  1024, 4, 1024//64, 1024,   2,   1,   1,   2,   2,  8,   True,  True,  False, False, None),
    (32,   1024,  1024, 8, 1024//64, 1024,   2,   1,   1,   2,   2,  8,   True,  True,  False, False, None),
],

8: [
    #B,   S,     H     L,   #head,   V,      LD0, LD1, PD0, PD1, PP, NB,  FM,    Remat, RS,    AP
    (64,   1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  16,  True,  True,  False, False, None), # 0.323
    #(64,   1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  16,  False, True,  False, False, None), # 0.380
    #(128,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  32,  True,  True,  False, False, None), # 0.323
    #(128,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  32,  False, True,  False, False, None), # 0.380
]
}

# yapf: enable

benchmark_suites = {
    "default": default_suite,
    "paper_gpt": paper_gpt_suite,
    "test_gpt": test_gpt_suite,
    "paper_auto_gpt": paper_auto_gpt_suite,
    "test_auto_gpt": test_auto_gpt_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=5)  # 2 warmup + 5 actual run.
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()),
                        default="paper_gpt")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--no-separate-process", action='store_false',
                        help="Do not launch separate processes for benchmark."
                             "Errors in a single case will terminate this script.",
                        dest='use_separate_process')
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    print(f"- Use separate process: {args.use_separate_process}")

    # Get the number of devices
    ray.init(address="auto")
    if args.num_hosts is not None or args.num_devices_per_host is not None:
        assert args.num_hosts is not None and args.num_devices_per_host is not None
        num_hosts, num_devices_per_host = args.num_hosts, args.num_devices_per_host
    else:
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
         use_remat, prefer_reduce_scatter, auto_pipeline, overwrite_global_config_dict) = benchmark_case
        model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads)

        if pipeline_mp_size <= 1 and not auto_pipeline:
            print(f"Skip the case: {str(benchmark_case)}, because PP <= 1. "
                  f"Please use `benchmark_gpt_bert_2d.py` "
                  f"since 3d runtime will have a small overhead.")
            continue

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(args.model, benchmark_case, args.niter,
                                    num_hosts, num_devices_per_host,
                                    use_separate_process=args.use_separate_process,
                                    disable_tqdm=args.disable_tqdm)
        (parameter_count, mem_allocated, max_mem_allocated, latencies, tflops,
         tflops_ckpt, compute_cost_file_name, forward_stage_layer_ids,
         submesh_shapes, logical_mesh_shapes, autosharding_global_configs) = result

        if not auto_pipeline:
            heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape",
                     "#Microbatch", "Force Mapping", "Remat", "Reduce-scatter",
                     "Mean Time", "Std Time", "#Params", "TFLOPs",
                     "TFLOPs (ckpt)", "Peak Mem", "overwrite_global_config_dict"]
            parallel_config = (l_dim0, l_dim1, pipeline_mp_size)
            values = [args.model, model_config, parallel_config, (p_dim0, p_dim1),
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
                     "Logical Mesh Shapes", "Autosharding Global Configs", "overwrite_global_config_dict"]
            values = [args.model + "-auto", model_config, num_gpus, pipeline_mp_size,
                      num_micro_batches, use_remat, prefer_reduce_scatter,
                      f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                      f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                      f"{max_mem_allocated/GB:.3f}G", compute_cost_file_name,
                      forward_stage_layer_ids, submesh_shapes,
                      logical_mesh_shapes, autosharding_global_configs, overwrite_global_config_dict]
            write_tsv(heads, values, output_name)

        time.sleep(0.1)  # for ctrl+c to work
