import sys
sys.path.append("../benchmark/alpa/")

import argparse
import time

from alpa.util import write_tsv, run_cmd

from benchmark.alpa.benchmark_3d_one_case import benchmark_one_case
from suite_artifact_e2e_gpt import (artifact_search_e2e_gpt_suite, artifact_result_e2e_gpt_suite)

benchmark_suites = {
    "gpt.search": artifact_search_e2e_gpt_suite,
    "gpt.result": artifact_result_e2e_gpt_suite,
}

method_name = {
    "gpt.search": "parax.auto",
    "gpt.result": "parax.auto",
}


def benchmark_one_suite(suite_name, num_hosts, num_devices_per_host, exp_name, output_name, instance="p3.16", niter=3,
                        use_separate_process=True, disable_tqdm=False):
    # Get the benchmark suite
    num_gpus = num_hosts * num_devices_per_host
    try:
        suite = benchmark_suites[suite_name][num_gpus]
    except KeyError:
        suite = None
    if not suite:
        print(f"No available benchmark suite for {suite_name} on {num_gpus} GPUs")
        exit()
    run_cmd("mkdir -p tmp")

    model_type = suite_name.split(".")[0]

    # Run all cases
    for benchmark_case in suite:
        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(model_type, benchmark_case, niter,
                                    num_hosts, num_devices_per_host,
                                    use_separate_process=use_separate_process,
                                    disable_tqdm=disable_tqdm)
        (parameter_count, mem_allocated, max_mem_allocated, latencies, tflops,
         tflops_ckpt, compilation_times, compute_cost_file_name, forward_stage_layer_ids,
         submesh_shapes, logical_mesh_shapes, autosharding_option_dicts) = result

        # Write to file
        heads = ["exp_name", "instance", "num_hosts", "num_devices_per_host", "model_name", "method", "value", "time_stamp"]
        values = [exp_name, instance, num_hosts, num_devices_per_host, model_type, method_name[suite_name],
                  str({"tflops": tflops_ckpt, "parameter_count": parameter_count / (10 ** 9)}), time.time()]
        write_tsv(heads, values, output_name)
        time.sleep(0.1)  # for ctrl+c to work


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--niter", type=int, default=3,
        help="The number of benchmark iterations")
    parser.add_argument("--no-separate-process", action='store_false',
                        help="Do not launch separate processes for benchmark."
                             "Errors in a single case will terminate this script.",
                        dest='use_separate_process')
    parser.add_argument("--instance", type=str, default="p3.16")
    parser.add_argument("--exp-name", type=str, default="e2e")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    cluster_sizes = [(4, 8), (2, 8), (1, 8), (1, 4), (1, 2), (1, 1)]
    output_name = f"results_{args.exp_name}.tsv"

    # GPT e2e results
    gpt_suite_name = "gpt.search" if args.search else "gpt.result"
    for num_hosts, num_devices_per_host in cluster_sizes:
        benchmark_one_suite(gpt_suite_name,
                            num_hosts,
                            num_devices_per_host,
                            args.exp_name,
                            output_name,
                            instance=args.instance,
                            niter=args.niter,
                            use_separate_process=args.use_separate_process,
                            disable_tqdm=args.disable_tqdm)
