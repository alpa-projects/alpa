import sys

sys.path.append("../benchmark/alpa/")

import argparse
import time
import numpy as np

from alpa.util import write_tsv, run_cmd

from benchmark.alpa.benchmark_2d_one_case import benchmark_one_case as benchmark_one_case_2d
from benchmark.alpa.benchmark_3d_one_case import benchmark_one_case
from suite_artifact_e2e_gpt import (artifact_search_e2e_gpt_suite, artifact_result_e2e_gpt_suite)
from suite_artifact_e2e_moe import (artifact_search_e2e_moe_suite, artifact_result_e2e_moe_suite)
from suite_artifact_e2e_wresnet import (artifact_search_e2e_wresnet_suite, artifact_result_e2e_wresnet_suite,
    artifact_search_e2e_wresnet_ppdp_suite, artifact_e2e_wresnet_intra_only_suite,
    artifact_search_e2e_wresnet_inter_only_suite, artifact_result_e2e_wresnet_ppdp_suite,
    artifact_result_e2e_wresnet_inter_only_suite)

benchmark_suites = {
    "gpt.search": artifact_search_e2e_gpt_suite,
    "gpt.result": artifact_result_e2e_gpt_suite,
    "moe.search": artifact_search_e2e_moe_suite,
    "moe.result": artifact_result_e2e_moe_suite,
    "wresnet.search": artifact_search_e2e_wresnet_suite,
    "wresnet.search.ppdp": artifact_search_e2e_wresnet_ppdp_suite,
    "wresnet.search.inter-only": artifact_search_e2e_wresnet_inter_only_suite,
    "wresnet.search.intra-only": artifact_e2e_wresnet_intra_only_suite,
    "wresnet.result": artifact_result_e2e_wresnet_suite,
    "wresnet.result.ppdp": artifact_result_e2e_wresnet_ppdp_suite,
    "wresnet.result.inter-only": artifact_result_e2e_wresnet_inter_only_suite,
    "wresnet.result.intra-only": artifact_e2e_wresnet_intra_only_suite,
}


def benchmark_one_suite(suite_name, num_hosts, num_devices_per_host, exp_name,
                        output_name, instance="p3.16", niter=3,
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
        if model_type == "wresnet" and suite_name.split(".")[-1] == "intra-only":
            result = benchmark_one_case_2d(model_type, benchmark_case, niter,
                                           num_hosts, num_devices_per_host,
                                           use_separate_process=use_separate_process,
                                           disable_tqdm=disable_tqdm)
            parameter_count, ilp_objective, peak_mem, latencies, tflops_ckpt = result
        else:
            result = benchmark_one_case(model_type, benchmark_case, niter,
                                        num_hosts, num_devices_per_host,
                                        use_separate_process=use_separate_process,
                                        disable_tqdm=disable_tqdm)
            (parameter_count, mem_allocated, max_mem_allocated, latencies, tflops,
            tflops_ckpt, compilation_times, compute_cost_file_name, forward_stage_layer_ids,
            submesh_shapes, logical_mesh_shapes, autosharding_option_dicts) = result

        # Write to file
        heads = ["exp_name", "instance", "num_hosts", "num_devices_per_host", "model_name", "method", "value", "time_stamp"]
        values = [exp_name, instance, num_hosts, num_devices_per_host, model_type, "parax.auto",
                  str({"tflops": tflops_ckpt, "parameter_count": parameter_count / (10 ** 9)}), time.time()]
        write_tsv(heads, values, output_name)
        time.sleep(0.1)  # for ctrl+c to work


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--niter", type=int, default=3,
        help="The number of benchmark iterations")
    parser.add_argument("--cluster-size", type=int)
    args = parser.parse_args()

    cluster_sizes = [(4, 8), (2, 8), (1, 8), (1, 4), (1, 1)]
    output_name = f"results_e2e.tsv"

    if args.cluster_size:
        cluster_sizes = [x for x in cluster_sizes if np.prod(x) == args.cluster_size]

    print(f"Cluster sizes: {cluster_sizes}")

    if args.model is None:
        models = ["gpt", "moe", "wresnet"]
    else:
        models = [args.model]

    for model in models:
        suite_name = model + (".search" if args.search else ".result")
        for num_hosts, num_devices_per_host in cluster_sizes:
            if model == "wresnet" and False:
                for search_space in ["ppdp", "inter-only", "intra-only"]:
                    suite_with_search_space = suite_name + ("." + search_space)
                    print("Running suite:", suite_with_search_space)
                    benchmark_one_suite(suite_with_search_space,
                                        num_hosts,
                                        num_devices_per_host,
                                        "e2e",
                                        output_name,
                                        instance="p3.16",
                                        niter=args.niter,
                                        use_separate_process=True,
                                        disable_tqdm=False)
            benchmark_one_suite(suite_name,
                                num_hosts,
                                num_devices_per_host,
                                "e2e",
                                output_name,
                                instance="p3.16",
                                niter=args.niter,
                                use_separate_process=True,
                                disable_tqdm=False)
