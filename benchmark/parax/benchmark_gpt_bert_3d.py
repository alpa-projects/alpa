import argparse

import ray

from benchmark.parax.benchmark_gpt_bert_3d_one_case import benchmark_one_case, setup_benchmark
from benchmark.util import run_cmd
from datetime import datetime

GB = 1024 ** 3


# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, LD0 = logical mesh dim 0, LD1 = logical mesh_dimension_1
# PD0 = physical mesh dim 0, PD = physical mesh dim 1
# FD = Force DP, NB = number of microbatches, Remat: rematerialization

# yapf: disable

sanity_check_suite = {

4: [
    # B,  S,     H,    L,  #head,    V   LD0, LD1, PD0, PD1, PP, NB,   FD,  Remat, Tie, Auto-layer-slicing
    (64, 1024, 1024, 4, 1024//64, 51200, 2, 1, 1, 2, 2, 1, True, True, False, False),
    (16, 1024, 1024, 4, 1024//64, 51200, 2, 1, 1, 2, 2, 2, True, True, False, False),
    (16, 1024, 1024, 4, 1024//64, 51200, 2, 1, 1, 2, 2, 8, True, True, False, False),
    (16, 1024, 1024, 4, 1024//64, 51200, 2, 1, 1, 2, 2, 4, True, True, False, False),
],

8: [
    # the performance below on p3.16
    # Parax: 0.602, 0.618, 0.543, 0.563
    # Megatron: 0.596 (DP), 0.69 (MP)
    (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   True, True, True),
    (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   False, True, True),
    (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   True, True, False),
    (32,  1024,  1024, 24, 1024//64, 51200, 4,   1,   1,   4,   2,  8,   False, True, False),
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
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--suite", choices=["default", "sanity_check"], default="sanity_check")
    parser.add_argument("--mode", choices=["normal", "nonstop"], default="normal")
    parser.add_argument("--output", type=str, default="result")
    args = parser.parse_args()

    print("- Benchmarking in {} mode.".format(args.mode))
    if args.mode == "normal":
        ray.init(address="auto")
        setup_benchmark()
        num_gpus = int(ray.cluster_resources()["GPU"])
        try:
            suite = benchmark_suites[args.suite][num_gpus]
        except KeyError:
            suite = None
        if not suite:
            print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
            exit()
        for case in suite:
            benchmark_one_case(case, args)
        ray.shutdown()
    elif args.mode == "nonstop":
        ray.init(address="auto")
        num_gpus = int(ray.cluster_resources()["GPU"])
        ray.shutdown()

        # construct case str
        output_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        for case in benchmark_suites[args.suite][num_gpus]:
            case_str = str(case)
            ret = run_cmd("python3 benchmark_gpt_bert_3d_one_case.py "
                         f"--model {args.model} "
                         f"--niter {args.niter} "
                         f'--case "{case_str}" '
                         f"--output {output_name}")
            # print("Exit code: {}".format(ret))
    else:
        raise RuntimeError()
