import argparse
from datetime import datetime

import numpy as np
import ray

from parax.util import write_tsv, run_cmd
from benchmark.parax.benchmark_gpt_bert_3d_one_case import benchmark_one_case
from benchmark.parax.paper_manual_gpt_suite import paper_gpt_suite, test_gpt_suite

GB = 1024 ** 3


# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, LD0 = logical mesh dim 0, LD1 = logical mesh_dimension_1
# PD0 = physical mesh dim 0, PD = physical mesh dim 1
# FD = Force DP, NB = number of microbatches, Remat: rematerialization

# yapf: disable

sanity_check_suite = {

4: [
    # B,  S,     H,    L,  #head,    V   LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, Tie, Auto-layer-slicing

    # NVprof case
    (32,  1024,  1024, 4, 1024//64, 1024, 2,   1,   1,   2,   2,  8,   True, True, False, False),
    (32,  1024,  1024, 8, 1024//64, 1024, 2,   1,   1,   2,   2,  8,   True, True, False, False),
],

8: [
    # the performance below on p3.16
    # Parax: 0.602 (DP + TIE), 0.618 (MP + TIE), 0.543 (DP + no-tie), 0.563 (MP + no-tie)
    # Megatron: 0.596 (DP), 0.69 (MP)
    (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   True, True, True, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   True, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   False, True, True, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   False, True, False, False),

    (64,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  16,   True, True, False, False), # 0.323
    (64,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  16,   False, True, False, False), # 0.380
    (128,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  32,   True, True, False, False), # 0.323
    (128,  1024,  1024, 12, 1024//64, 51200, 4,   1,   1,   4,   2,  32,   False, True, False, False), # 0.380
]
}

default_benchmark_suite = {

8: [
    # B,  S,     H,    L,  #head,    V      LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, Tie, Auto-layer-slicing

    (128,  512,  1024, 10, 1024//64,  25600, 1,  4, 1, 4,  2,  32,  True, True),

    # GPT-2 355M, DP + PP2, single node 8 GPUs, w/o remat
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  1,   True, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  2,   True, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  4,   True, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  8,   True, False, False, False),

    # GPT-2 355M, DP + PP2, single node 8 GPUs, with remat
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  1,   True,  True, False, False), # OOM
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  2,   True,  True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  4,   True,  True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  8,   False, True, False, False),

    # GPT-2 355M, auto sharding (best of [DP, MP]) + PP2, w/o remat
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  1,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  2,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  4,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  8,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  16,  False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  32,  False, False, False, False),

    # GPT-2 355M, auto sharding (best of [DP, MP]) + PP2, with remat
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  1,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  2,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  4,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  8,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  16,  False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  32,  False, True, False, False),

    # GPT-3 355M, DP + PP4, w/o remat
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  1,   True, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  2,   True, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  4,   True, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  8,   True, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  16,  True, False, False, False),

    # GPT-3 355M, DP + PP4, w/ remat
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  1,   True, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  2,   True, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  4,   True, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  8,   True, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  16,  True, True, False, False),

    # GPT-2 355M, auto sharding (best of [DP, MP]) + PP4
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  1,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  2,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  4,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  8,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  16,  False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  32,  False, False, False, False),

    # GPT-2 355M, auto sharding (best of [DP, MP]) + PP4, w/ remat
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  1,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  2,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  4,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  8,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  16,  False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   2,   1,   2,   4,  32,  False, True, False, False),

    # GPT-3 355M, PP8
    # (16,  1024,  1024, 24, 1024//64,  51200, 2,  2,  2,  8,  False, False),  # sanity check case
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  1,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  2,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  4,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  8,   False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  16,  False, False, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  32,  False, False, False, False),

    # GPT-3 355m, PP8, w/ remat
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  1,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  2,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  4,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  8,   False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  16,  False, True, False, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 1,   1,   1,   1,   8,  32,  False, True, False, False),

    # # GPT-2 355M, auto sharding (best of [DP, MP]) + PP2, with remat, with auto layer
    # # When auto layer is on, pipeline_mp_size will used to set the number of
    # # layers for auto layer slicing
    # (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  1,   False, True, True, False),
    # (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  2,   False, True, True, False),
    # (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  4,   False, True, True, False),
    # (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  8,   False, True, True, False),
    # (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  16,  False, True, True, False),
    # (32,  1024,  1024, 24, 1024//64, 51200, 1,   4,   1,   4,   2,  32,  False, True, True, False),

    # GPT-2 355M, auto sharding (best of [DP, MP]) + PP2, with remat, with auto layer & auto stage
    # (32,  1024,  1024, 6, 1024//64, 51200, 0,   0,   0,   0,   8,  4,   False, True, True, True),
],

16: [
    # B,  S,    H,    L,  #head,     V,     DP, TP, PP, NB, FD, RS
]
}

# yapf: enable

benchmark_suites = {
    "default": default_benchmark_suite,
    "sanity_check": sanity_check_suite,
    "paper_gpt": paper_gpt_suite,
    "test_gpt": test_gpt_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=7)  # 2 warmup + 5 actual run.
    parser.add_argument("--suite", choices=["default", "sanity_check", "paper_gpt", "test_gpt"],
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
    output_name = f"{args.model}_parax_{args.exp_name}-{date_str}.tsv"
    for case in suite:
        dp, mp, pp = case[6], case[7], case[10]
        if pp <= 1:
            print(f"Skipping the case: {str(case)}, because PP <= 1. Lianmin will test it.")
            continue

        result = benchmark_one_case(args.model, case, args.niter,
                                    use_separate_process=args.use_separate_process)
        parameter_count, mem_allocated, max_mem_allocated, latencies, tflops, tflops_ckpt = result

        heads = ["Type", "Model Config", "Parallel Config", "P-mesh shape", "#Microbatch",
                 "Force DP", "Remat", "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem",]
        paralell_config = (dp, mp, pp)
        values = [args.model, str(case[:5]), str(paralell_config), str(case[8:10]),
                  str(case[11]), str(case[12]), str(case[13]),
                  f"{np.mean(latencies):.3f}", f"{np.std(latencies):.3f}",
                  f"{parameter_count/1e9:.3f}", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                  f"{max_mem_allocated/GB:.3f}"]
        write_tsv(heads, values, output_name)
