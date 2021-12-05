import argparse
from datetime import datetime

import numpy as np
import ray

from parax.util import write_tsv, run_cmd
from benchmark.parax.benchmark_gpt_bert_3d_one_case import benchmark_one_case
from benchmark.parax.paper_manual_gpt_suite import paper_gpt_suite, test_gpt_suite
from benchmark.parax.paper_auto_gpt_suite import paper_auto_gpt_suite, test_auto_gpt_suite
from parax.pipeline_parallel.stage_construction import get_last_dp_result

GB = 1024 ** 3

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, LD0 = logical_mesh_dimension_0, LD1 = logical_mesh_dimension_1,
# PD0 = physical_mesh_dimension_0, PD1 = physical_mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping, Remat = use_rematerialization
# RS = prefer_reduce_scatter

# yapf: disable

default_suite = {
4: [
    #B,   S,     H     L,   #head,   V,      LD0, LD1, PD0, PD1, PP, NB,  FM,    Remat, RS,    Auto-pipeline
    (32,   1024,  1024, 4, 1024//64, 1024,   2,   1,   1,   2,   2,  8,   True,  True,  False, False),
    (32,   1024,  1024, 8, 1024//64, 1024,   2,   1,   1,   2,   2,  8,   True,  True,  False, False),
],

8: [
    #B,   S,     H     L,   #head,   V,      LD0, LD1, PD0, PD1, PP, NB,  FM,    Remat, RS,    Auto-pipeline
    (64,   1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  16,  True,  True,  False, False), # 0.323
    (64,   1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  16,  False, True,  False, False), # 0.380
    (128,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  32,  True,  True,  False, False), # 0.323
    (128,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  32,  False, True,  False, False), # 0.380
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
    parser.add_argument("--niter", type=int, default=7)  # 2 warmup + 5 actual run.
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()),
                        default="paper_gpt")
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

    # Run all cases
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"{args.model}_parax_{args.exp_name}_{date_str}.tsv"
    for case in suite:
        dp, mp, pp = case[6], case[7], case[10]
        auto_layer_and_stage = case[15]
        if pp <= 1 and not auto_layer_and_stage:
            print(f"Skipping the case: {str(case)}, because PP <= 1. Please use `benchmark_gpt_bert_2d.py` "
                  f"since 3d will have a small overhead.")
            continue
        print("Working on case: {}".format(str(case)))
        result = benchmark_one_case(args.model, case, args.niter,
                                    use_separate_process=args.use_separate_process)
        (parameter_count, mem_allocated, max_mem_allocated, latencies, tflops,
         tflops_ckpt, compute_cost_file_name, forward_stage_layer_ids,
         submesh_shapes, logical_mesh_shapes, autosharding_global_configs) = result

        if not auto_layer_and_stage:
            heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape",
                     "#Microbatch", "Force DP", "Remat", "Reduce-scatter",
                     "Mean Time", "Std Time", "#Params", "TFLOPs",
                     "TFLOPs (ckpt)", "Peak Mem",]
            parallel_config = (dp, mp, pp)
            values = [args.model, str(case[:6]), str(parallel_config), str(case[8:10]),
                      str(case[11]), str(case[12]), str(case[13]), str(case[14]),
                      f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                      f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                      f"{max_mem_allocated/GB:.3f}G"]
            write_tsv(heads, values, output_name)
        else:
            heads = ["Type", "Model Config", "#GPUs", "#Layers (for Auto-Layer)",
                     "#Microbatch", "Remat", "Reduce-scatter",
                     "Mean Time", "Std Time", "#Params", "TFLOPs",
                     "TFLOPs (ckpt)", "Peak Mem", "Compute Cost File",
                     "Layer->Stage Mapping", "Submesh Shapes",
                     "Logical Mesh Shapes", "Autosharding Global Configs"]
            values = [args.model + "-auto", str(case[:6]), str(num_gpus), str(pp),
                      str(case[11]), str(case[13]), str(case[14]),
                      f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                      f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                      f"{max_mem_allocated/GB:.3f}G", str(compute_cost_file_name),
                      str(forward_stage_layer_ids), str(submesh_shapes),
                      str(logical_mesh_shapes), str(autosharding_global_configs)]
            write_tsv(heads, values, output_name)

