"""The entry point of intra-op + inter-op parallelism benchmark."""
import argparse
from datetime import datetime
import time

import numpy as np

from alpa.util import write_tsv, run_cmd, get_num_hosts_and_num_devices, GB

from benchmark_3d_infer_one_case import benchmark_one_case
import suite_gpt_infer

benchmark_suites = {
    "gpt.infer_perf_test": suite_gpt_infer.perf_test_suite,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        choices=list(benchmark_suites.keys()),
                        type=str,
                        required=True)
    parser.add_argument("--niter",
                        type=int,
                        default=3,
                        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--no-separate-process",
                        action='store_false',
                        help="Do not launch separate processes for benchmark."
                        "Errors in a single case will terminate this script.",
                        dest='use_separate_process')
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--result_name", type=str, default="result")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--stream-mode", action="store_true")
    args = parser.parse_args()

    # Get the benchmark suite
    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)
    num_gpus = num_hosts * num_devices_per_host

    try:
        suite = benchmark_suites[args.suite][(num_hosts, num_devices_per_host)]
    except KeyError:
        suite = None
    if not suite:
        print(
            f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()
    run_cmd("mkdir -p tmp")

    model_type = args.suite.split(".")[0]
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_name = f"{args.result_name}.tsv"

    # Run all cases
    for benchmark_case in suite:
        if model_type in ["gpt"]:
            (model_name, no_embedding, batch_size, seq_len, hidden_size,
             num_layers, num_heads, vocab_size, num_micro_batches,
             parallel_mode, parallel_args) = benchmark_case
            model_config = (batch_size, seq_len, hidden_size, num_layers,
                            num_heads)
        else:
            raise ValueError(f"Invalid model: {model_type}")

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(
            model_type,
            benchmark_case,
            args.niter,
            num_hosts,
            num_devices_per_host,
            use_separate_process=args.use_separate_process,
            disable_tqdm=args.disable_tqdm,
            stream_mode=args.stream_mode)
        (parameter_count, max_mem_allocated, overall_latency, e2e_latency,
         tflops, compilation_times, compute_cost_file_name,
         forward_stage_layer_ids, submesh_shapes, logical_mesh_shapes,
         autosharding_option_dicts) = result

        heads = [
            "Type",
            "Model Size",
            "BatchSize",
            "#MicroBatch",
            "TotalBatchSize",
            "#Hosts",
            "#GPUsPerHost",
            "#PP",
            "#DP",
            "#OP",
            "E2E Latency",
        ]
        if args.stream_mode:
            heads += [
                "niter",
            ]
        else:
            heads += [
                "Ray Overhead",
                "Overall Latency",
                "TFLOPs",
                "Peak Mem",
            ]

        values = [
            model_type,
            model_name,
            batch_size // num_micro_batches,
            num_micro_batches,
            batch_size,
            num_hosts,
            num_gpus,
            parallel_args[2],
            *parallel_args[3][2][0],
            f"{e2e_latency:.3f}",
        ]

        if args.stream_mode:
            values += [
                args.niter,
            ]
        else:
            values += [
                f"{(e2e_latency - overall_latency):.3f}",
                f"{overall_latency:.3f}",
                f"{tflops:.2f}",
                f"{max_mem_allocated/GB:.3f}",
            ]

        write_tsv(heads, values, result_name)

        time.sleep(0.1)  # for ctrl+c to work
