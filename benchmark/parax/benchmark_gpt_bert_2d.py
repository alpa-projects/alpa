import argparse

import ray

import numpy as np

from parax.util import list_gpu_info, write_tsv

from benchmark_gpt_bert_2d_one_case import benchmark_one_case

GB = 1 << 30

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel,
# RS = prefer_reduce_scatter, CK = use_checkpoint

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    # B,  S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    #(16,  512,  1024, 10, 1024//64,  25600, 1,  1,  1,  False, False, False),
    (8,  1024,  1536, 10, 1536//96,  25600, 1,  1,  1,  False, False, True),
],

4: [
    # B,   S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
],

8: [
    # B,   S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    (256,  512,  1024, 10, 1024//64,  25600, 8,  1,  1,  False, True,  False),
    (512,  512,  1024, 10, 1024//64,  25600, 8,  1,  2,  False, True,  False),
    (8,    1024, 4096, 10, 4096//128, 25600, 8,  1,  1,  True,  True,  False),
    (8,    1024, 4096, 10, 4096//128, 25600, 2,  4,  1,  False, True,  False),
    (8,    1024, 4096, 10, 4096//128, 25600, 1,  8,  1,  False, True,  False),
    (8,    1024, 4096, 10, 4096//128, 25600, 1,  8,  1,  False, True,  True),
],

16: [
    # B,   S,    H,    L,  #head,     V,     D0, D1, NB, FD,    RS,    CK
    #(512,  512,  1024, 10, 1024//64,  25600, 16, 1,  1,  False, True,  False),
    #(2048, 512,  1024, 10, 1024//64,  25600, 16, 1,  4,  False, True,  False),
    #(16,   1024, 4096, 10, 4096//128, 25600, 2,  8,  1,  False, True,  False),
    #(64,   1024, 4096, 10, 4096//128, 25600, 2,  8,  4,  False, True,  False),
    #(16,   1024, 4096, 10, 4096//128, 25600, 16, 1,  1,  False, True,  False),
    #(64,   1024, 4096, 10, 4096//128, 25600, 16, 1,  4,  False, True,  False),

    (16,    1024, 6144, 10, 6144//128, 25600, 2,  8,  1,  False, False, False),
    (64,    1024, 6144, 10, 6144//128, 25600, 2,  8,  4,  False, False, False),
    (128,   1024, 6144, 10, 6144//128, 25600, 2,  8,  8,  False, False, False),
    (16,    1024, 6144, 10, 6144//128, 25600, 2,  8,  1,  False, True,  False),
    (64,    1024, 6144, 10, 6144//128, 25600, 2,  8,  4,  False, True,  False),
    (128,   1024, 6144, 10, 6144//128, 25600, 2,  8,  4,  False, True,  False),
]
}

benchmark_suites = {
    "default": default_benchmark_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=10,
        help="Number of benchmark iteration")
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--suite", choices=["default"], default="default")
    parser.add_argument("--use-separate-process", action="store_true",
        help="Launch separate processes for benchmark to isolate errors."
              "Erros in a single case will not terminate this script.")
    args = parser.parse_args()

    # Get benchmark suite and run all cases
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        ray.init(address="auto")
        num_gpus = int(ray.cluster_resources()["GPU"])
    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None
    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()

    # Run all cases
    for case in suite:
        result = benchmark_one_case(args.model, case, args.niter, args.local,
                                    args.use_separate_process)
        param_count, ilp_objective, alloc_mem, latencies, tflops = result

        # Log results
        heads = ["Model", "Model Config", "Parallel Config", "Param Count",
                 "Alloc Mem", "ILP Objective", "Mean Latency", "Std Latency", "TFLOPS"]
        values = [args.model, case[:-6], case[-6:],
                  f"{param_count/1e9:.3f}", f"{alloc_mem/GB:.3f}", f"{ilp_objective:.2f}",
                  f"{np.mean(latencies):.3f}", f"{np.std(latencies):.3f}", f"{tflops:.2f}"]
        write_tsv(heads, values, f"result_{args.model}.tsv")
