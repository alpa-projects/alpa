import argparse
from datetime import datetime
import time
from benchmark.parax.benchmark_wide_resnet_3d_one_case import benchmark_wresnet_3d_one_case

import numpy as np
import ray
from parax.util import (run_cmd, write_tsv)

# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor,
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel,
# RS = prefer_reduce_scatter, CK = use_checkpoint

# benchmark suite for GPUs with 32GB memory
benchmark_suite_32gb = {  # key = number of gpus, value = a list of cases
    1: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    CK,
    (32,   224, 50,  256, 4, "fp32", 1,  1,  1,  False, True,  False),
    ],

    4: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    CK,
    #(32,   224, 50,  512, 2, "fp32", 1,  4,  1,  False, False, False),
    ],

    8: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    CK,
    #(256,  224, 50,  256, 4, "fp32", 8,  1,  1,  False, True,  False),
    #(64,   224, 50,  512, 4, "fp32", 2,  4,  1,  False, True,  False),
    #(128,  224, 50,  512, 4, "fp32", 2,  4,  2,  False, True,  False),
    #(32,   224, 50,  704, 4, "fp32", 8,  1,  1,  False, True,  False),

    #(64,    224, 50,  512, 2, "fp32", 2,  4,  1,  False, False, False),
    (128,   224, 50,  512, 2, "fp32", 2,  4,  2,  False, False, False),
    ],

    16: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    CK,
    (64,   224, 50,  576, 4, "fp32", 2,  8,  1,  False, False, False),
    (256,  224, 50,  576, 4, "fp32", 2,  8,  4,  False, False, False),
    (512,  224, 50,  576, 4, "fp32", 2,  8,  8,  False, False, False),

    (64,   224, 50,  576, 4, "fp32", 2,  8,  1,  False, True,  False),
    (256,  224, 50,  576, 4, "fp32", 2,  8,  4,  False, True,  False),
    (512,  224, 50,  576, 4, "fp32", 2,  8,  8,  False, True,  False),
    ],

}

# benchmark suite for GPUs with 16GB memory
benchmark_suite_16gb = {  # key = number of gpus, value = a list of cases

    1: [
    #   B,   I,  L,   C,   W,  dtype, NB,     FD,   RS,    CK,
    (2048, 224, 50,  160,  2, "fp32", 32,  False, False,  True),
    ],

    2: [
    #   B,   I,  L,   C,   W,  dtype, NB,     FD,   RS,    CK,
    (1536, 224, 50,  224,  2, "fp32", 32,  False, True,  True),
    ],

    4 : [
    #   B,   I,  L,   C,   W,  dtype, NB,     FD,    RS,    CK,
    (1536, 224, 50,  320,  2, "fp32", 32,  False,  True,  True),
    ],

    8: [
    #   B,   I,   L,   C,   W,  dtype, NB,     FD,    RS,    CK,
    (1536, 224, 50,  448,   2, "fp32", 32,  False,  True,  True),
    ],

    16: [
    #   B,   I,   L,   C,   W,  dtype, NB,     FD,   RS,    CK,
    (1536, 224, 50,  640,   2, "fp32", 32,  False, True,  True),
    ],

    32: [
    #   B,   I,   L,   C,   W,  dtype, NB,     FD,   RS,    CK,
    (1536, 224, 50,  640,   2, "fp32", 32,  False, True,  True),
    (1520, 224, 50,  320,  16, "fp32", 38,  False, False,  True),
    ],

}

benchmark_suites = {
    "32gb": benchmark_suite_32gb,
    "16gb": benchmark_suite_16gb,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--niter",
                        type=int,
                        default=4,
                        help="Number of benchmark iteration")
    parser.add_argument("--suite",
                        choices=["32gb", "16gb"],
                        default="16gb",
                        help="The benchmark suite")
    parser.add_argument(
        "--logical_mesh_search_space",
        choices=["single_node_model_parallel", "only_dp", "all", "default"],
        default="single_node_model_parallel",
        help="logical mesh search space in auto stage construction")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()

    # Set global environments
    ray.init(address="auto")
    if (args.num_hosts is not None or args.num_devices_per_host is not None):
        assert (args.num_hosts is not None and
                args.num_devices_per_host is not None)
        num_gpus = args.num_hosts * args.num_devices_per_host
    else:
        num_gpus = int(ray.cluster_resources()["GPU"])

    # Get benchmark suite and run all cases
    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None

    if not suite:
        print(
            f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()
    run_cmd("mkdir -p tmp")

    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"wide_resnet_parax_{args.exp_name}_{date_str}.tsv"

    for benchmark_case in suite:
        result = benchmark_wresnet_3d_one_case(
            benchmark_case,
            args.niter,
            num_hosts=args.num_hosts,
            num_devices_per_host=args.num_devices_per_host,
            logical_mesh_search_space=args.logical_mesh_search_space)
        latencies, peak_mem, tflops, param_count = result

        # Log results
        heads = [
            "Model", "Model Config", "Force Data Parallel", "Reduce Scatter",
            "Remat", "Param Count", "Peak Mem", "Mean Latency", "Std Latency",
            "TFLOPS"
        ]
        values = [
            "wide-resnet", benchmark_case[:-3], benchmark_case[-3],
            benchmark_case[-2], benchmark_case[-1], f"{param_count/1e9:.3f}",
            f"{peak_mem}", f"{np.mean(latencies):.3f}",
            f"{np.std(latencies):.3f}", f"{tflops:.2f}"
        ]
        print("write to ", output_name)
        write_tsv(heads, values, output_name)
        time.sleep(0.1)
