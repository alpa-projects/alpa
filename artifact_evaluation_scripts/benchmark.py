"""The entry point of intra-op + inter-op parallelism benchmark."""
import os
import argparse
from datetime import datetime
import time

import numpy as np

from alpa.util import (write_tsv, get_num_hosts_and_num_devices, to_str_round,
                       GB)

from benchmark_one_case import benchmark_one_case
import suite_manual_gpt
import suite_unet

benchmark_suites = {
    "gpt": suite_manual_gpt.perf_test_suite,
    "unet": suite_unet.perf_test_auto_suite
}


def benchmark_suite(suite_name,
                    comm_overlap_level,
                    num_hosts,
                    num_devices_per_host,
                    exp_name="default",
                    niter=3,
                    profile_driver_time=False,
                    profile_stage_execution_time=False,
                    disable_tqdm=False,
                    use_separate_process=True):
    num_gpus = num_hosts * num_devices_per_host

    if num_gpus not in benchmark_suites[suite_name]:
        print(f"No benchmark suite for #gpu={num_gpus}")
        return
    suite = benchmark_suites[suite_name][num_gpus]

    os.makedirs("tmp", exist_ok=True)

    model_type = suite_name.split(".")[0]
    output_name = f"{exp_name}.tsv"

    # Run all cases
    for benchmark_case in suite:
        model_config = benchmark_case.model_config
        num_micro_batches = benchmark_case.num_micro_batches
        parallel_args = benchmark_case.parallel_args

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(
            model_type,
            benchmark_case,
            comm_overlap_level,
            niter,
            num_hosts,
            num_devices_per_host,
            profile_driver_time=profile_driver_time,
            profile_stage_execution_time=profile_stage_execution_time,
            disable_tqdm=disable_tqdm,
            use_separate_process=use_separate_process)

        (parameter_count, peak_mem, latencies, tflops, metadata) = result

        heads = [
            "Type", "Model Config", "#Microbatch", "#GPU", "Parallel Config",
            "Mean Time (s)", "Std Time (s)", "#Params (Billion)", "TFLOPs",
            "Peak Mem (GB)", "Metadata"
        ]
        values = [
            model_type, model_config, num_micro_batches, num_gpus,
            parallel_args, f"{np.mean(latencies):.3f}",
            f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
            f"{tflops:.2f}", f"{peak_mem/GB:.3f}",
            to_str_round(metadata, 2)
        ]
        write_tsv(heads, values, output_name)

        time.sleep(0.1)  # for ctrl+c to work


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        choices=list(benchmark_suites.keys()),
                        type=str,
                        required=True)
    parser.add_argument(
        "--comm-overlap-level",
        choices=[0, 1, 2, 3],
        type=int,
        required=True,
        help="Level 0: no overlap between communication and computation; "
        "Level 1: overlap communication and computation; "
        "Level 2: use overlap friendly pipeline schedule; "
        "Level 3: hypothetical upper bound with signal send/recv")
    parser.add_argument("--niter",
                        type=int,
                        default=3,
                        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--profile-driver-time",
                        action="store_true",
                        help="Profile the execution time on the driver instead "
                        "of the workers.")
    parser.add_argument(
        "--profile-stage-execution-time",
        action="store_true",
        help="Profile the execution timestamps of each pipeline "
        "stage")
    parser.add_argument("--no-separate-process",
                        action="store_false",
                        help="Do not launch separate processes for benchmark. "
                        "Errors in a single case will terminate this "
                        "script.",
                        dest="use_separate_process")
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)

    benchmark_suite(args.suite, args.comm_overlap_level, num_hosts,
                    num_devices_per_host, args.exp_name, args.niter,
                    args.profile_driver_time, args.profile_stage_execution_time,
                    args.disable_tqdm, args.use_separate_process)
