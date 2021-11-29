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

gpt_specs = {
        # S，    H，   L,   #head, V,
"125M": (1024, 768,   12,  12,    51200),
"350M": (1024, 1024,  24,  16,    51200),
"760M": (1024, 1536,  24,  16,    51200),
"1.3B": (1024, 2048,  24,  32,    51200),
"2.7B": (1024, 2560,  32,  32,    51200),
"4.1B": (1024, 3200,  32,  32,    51200),
"6.7B": (1024, 4096,  32,  32,    51200),
"13B":  (1024, 5120,  40,  40,    51200),
"39B":  (1024, 8192,  48,  64,    51200),
"76B":  (1024, 10240, 60,  80,    51200),
}

paper_gpt_2d_suite = {  # key = number of gpus, value = a list of cases
1: [
    # B,   S, H, L, #head,  V,  D0, D1, NB, FD,    RS,    CK
    (24,   *gpt_specs["125M"],  1,  1,  1,  False, False, True),
    (32,   *gpt_specs["125M"],  1,  1,  1,  False, False, True),
    (40,   *gpt_specs["125M"],  1,  1,  1,  False, False, True),

    (16,   *gpt_specs["350M"],  1,  1,  1,  False, False, True),
    (24,   *gpt_specs["350M"],  1,  1,  1,  False, False, True),
    (32,   *gpt_specs["350M"],  1,  1,  1,  False, False, True),
],

2: [
    # B,   S, H, L, #head,  V,  D0, D1, NB, FD,    RS,    CK
    (32,   *gpt_specs["350M"],  1,  2,  1,  False, False, True),
    (40,   *gpt_specs["350M"],  1,  2,  1,  False, False, True),
    (48,   *gpt_specs["350M"],  1,  2,  1,  False, False, True),

    (8,    *gpt_specs["760M"],  1,  2,  1,  False, False, True),
    (16,   *gpt_specs["760M"],  1,  2,  1,  False, False, True),
],

4: [
    # B,   S, H, L, #head,  V,  D0, D1, NB, FD,    RS,    CK
    (40,   *gpt_specs["760M"],  2,  2,  1,  False, False, True),
    (48,   *gpt_specs["760M"],  2,  2,  1,  False, False, True),
    (24,   *gpt_specs["760M"],  1,  4,  1,  False, False, True),  # BUG
    (32,   *gpt_specs["760M"],  1,  4,  1,  False, False, True),

    (8,    *gpt_specs["1.3B"],  2,  2,  1,  False, False, True),
    (16,   *gpt_specs["1.3B"],  2,  2,  1,  False, False, True),
    (24,   *gpt_specs["1.3B"],  2,  2,  1,  False, False, True),
    (8,    *gpt_specs["1.3B"],  1,  4,  1,  False, False, True),  # BUG
    (16,   *gpt_specs["1.3B"],  1,  4,  1,  False, False, True),
],

8: [
    # B,   S, H, L, #head,  V,  D0, D1, NB, FD,    RS,    CK
    (24,   *gpt_specs["1.3B"],  4,  2,  1,  False, False, True),
    (32,   *gpt_specs["1.3B"],  4,  2,  1,  False, False, True),
    (40,   *gpt_specs["1.3B"],  2,  4,  1,  False, False, True),
    (48,   *gpt_specs["1.3B"],  2,  4,  1,  False, False, True),
    (40,   *gpt_specs["1.3B"],  1,  8,  1,  False, False, True),  # BUG
    (48,   *gpt_specs["1.3B"],  1,  8,  1,  False, False, True),

    (4,    *gpt_specs["2.7B"],  4,  2,  1,  False, False, True),
    (8,    *gpt_specs["2.7B"],  2,  4,  1,  False, False, True),
    (16,   *gpt_specs["2.7B"],  2,  4,  1,  False, False, True),
    (16,   *gpt_specs["2.7B"],  1,  8,  1,  False, False, True),  # BUG
    (24,   *gpt_specs["2.7B"],  1,  8,  1,  False, False, True),
],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--niter", type=int, default=5,
        help="Number of benchmark iteration")
    args = parser.parse_args()

    # Get benchmark suite and run all cases
    ray.init(address="auto")
    num_gpus = int(ray.cluster_resources()["GPU"])
    try:
        suite = paper_gpt_2d_suite[num_gpus]
    except KeyError:
        suite = None
    if not suite:
        print(f"No available benchmark suite for {num_gpus} GPUs")
        exit()

    # Run all cases
    for case in suite:
        result = benchmark_one_case(args.model, case, args.niter,
                                    local=False,
                                    use_separate_process=True)
        param_count, ilp_objective, peak_mem, latencies, tflops = result

        # Log results
        heads = ["Model", "Model Config", "Parallel Config", "Param Count",
                 "Peak Mem", "ILP Objective", "Mean Latency", "Std Latency", "TFLOPS"]
        values = [args.model, case[:-6], case[-6:],
                  f"{param_count/1e9:.3f}", f"{peak_mem/GB:.3f}", f"{ilp_objective:.2f}",
                  f"{np.mean(latencies):.3f}", f"{np.std(latencies):.3f}", f"{tflops:.2f}"]
        write_tsv(heads, values, f"result_{args.model}.tsv")
