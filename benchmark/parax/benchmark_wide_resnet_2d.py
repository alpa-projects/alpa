import argparse
from datetime import datetime

import ray
import numpy as np

from parax.util import list_gpu_info, write_tsv, run_cmd, get_num_hosts_and_num_devices
from benchmark_wide_resnet_2d_one_case import benchmark_one_case

GB = 1 << 30
_ = None

# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor, 
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FM = force_batch_dim_mapping,
# RS = prefer_reduce_scatter, Remat = use_rematerialization

default_benchmark_suite = {
1: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    Remat,
    (16,   224, 50,  192, 2, "fp32", 1,  1,  1,  False, True,  False),
],

4 : [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    Remat,
    (32,   224, 50,  320, 2, "fp32", 1,  4,  1,  False, False, False),
],

8: [
    #B,    I,   L,   C,   W, dtype,  D0, D1, NB, FD,    RS,    Remat,
    # data-parallel
    #(128,  224, 50,  192, 2, "fp32", 8,  1,  1,  True,  True,  False),
    #(512,  224, 50,  192, 2, "fp32", 8,  1,  4,  True,  True,  False),
    #(128,  224, 50,  192, 2, "fp32", 8,  1,  1,  False, True,  False),
    #(512,  224, 50,  192, 2, "fp32", 8,  1,  4,  False, True,  False),

    # model-parallel
    #(16,   224, 50,  320, 2, "fp32", 8,  1,  1,  True,   True,  False),
    #(64,   224, 50,  320, 2, "fp32", 8,  1,  4,  True,   True,  False),
    #(16,   224, 50,  320, 2, "fp32", 8,  1,  1,  False,  True,  False),
    #(64,   224, 50,  320, 2, "fp32", 8,  1,  4,  False,  True,  False),

    # 2d mesh
    (64,   224, 50,  320, 2, "fp32", 1,  8,  1,  False, False, False),
],
}

benchmark_suites = {
    "default": default_benchmark_suite,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=5,
        help="Number of benchmark iteration")
    parser.add_argument("--num-hosts", type=int)
    parser.add_argument("--num-devices-per-host", type=int)
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()), default="default")
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

    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"w_resnet_parax_{args.exp_name}_{date_str}.tsv"

    # Run all cases
    for benchmark_case in suite:
        batch_size, image_size, num_layers, num_channels, width_factor, dtype,\
            mesh_dim0, mesh_dim1, num_micro_batches, force_batch_dim_mapping,\
            prefer_reduce_scatter, use_remat = benchmark_case

        model_config = (batch_size, image_size, num_layers, num_channels, width_factor)
        parallel_config = (mesh_dim0, mesh_dim1)

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(benchmark_case, args.niter,
                                    num_hosts, num_devices_per_host,
                                    args.local, args.use_separate_process)
        param_count, ilp_objective, peak_mem, latencies, tflops = result

        # Log results
        heads = ["Type", "Model Config", "Parallel Config",
                 "#Microbatch", "Force Mapping", "Remat", "Reduce-scatter",
                 "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem", "ILP objective"]
        values = ["w-resnet", model_config, parallel_config, num_micro_batches,
                  force_batch_dim_mapping, use_remat, prefer_reduce_scatter,
                  f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{param_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops:.2f}",
                  f"{peak_mem/GB:.3f}G", f"{ilp_objective:.2f}" ]
        write_tsv(heads, values, output_name)
